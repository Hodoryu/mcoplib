"""Unit test + bandwidth benchmark for
   torch.ops.sgl_kernel.per_token_cast_to_fp8  on MetaX C600-U.

Op signature (in-place):
  per_token_cast_to_fp8(Tensor! out, Tensor! scale, Tensor input,
                        bool trans_scale=False) -> ()

Semantics (per group of 128 contiguous elements along the last dim):
  absmax_g = max_i |input[g*128 + i]|            (floored at 1e-4)
  scale[g] = absmax_g / 448
  out[g*128 + i] = quant_fp8( input[..] * 448/absmax_g )
  trans_scale=True  -> scale laid out as [G, m] (column-major w.r.t. [m, G])

Coverage:
  * every launch path: small(VEC8, non-NT) / VEC16 / VEC8 / VEC4,
    trans_scale scatter path (G<8 or m<32) and workspace+transpose path
  * bf16 and fp16 inputs
  * special values: zeros, below-floor tiny values, large values, outliers,
    all-negative, constant groups, odd m / tail blocks
  * out-of-bounds write detection (sentinel guard regions)
  * determinism
  * argument validation (n%128, dtypes)
  * bandwidth benchmark (trans_scale False / True)

Run:
  python unit_test/test_op_sglang_per_token_cast_to_fp8.py            # all
  python unit_test/test_op_sglang_per_token_cast_to_fp8.py --no-bench # tests only
  pytest -q unit_test/test_op_sglang_per_token_cast_to_fp8.py         # pytest
"""

import argparse
import os
import re
import subprocess
import sys
import time

import torch
import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)


# ------------------------- idle GPU auto-selection ---------------------------
def _auto_select_idle_gpus(max_gpus=2):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.check_output(["mx-smi"], text=True, timeout=30)
    except Exception:
        return None
    idle, cur_gpu = [], None
    for line in out.splitlines():
        gm = re.search(r"\|\s*(\d+)\s+MetaX C600", line)
        if gm:
            cur_gpu = int(gm.group(1))
        sm = re.search(r"\|\s+(\d+)\s+(On|Off)\s+\|", line)
        if sm:
            cur_gpu = int(sm.group(1))
        if "Available" in line and "0%" in line and cur_gpu is not None:
            if cur_gpu not in idle:
                idle.append(cur_gpu)
    if not idle:
        return None
    sel = ",".join(str(x) for x in idle[:max_gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = sel
    return sel


_auto_select_idle_gpus()

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
GROUP = 128
QMAX = 448.0
FP8_MIN_ABSMAX = 1e-4
COS_THRESHOLD = 0.999
SCALE_RTOL = 1e-6
FP8_MATCH_RATIO = 0.99          # >=99% of fp8 codes must match bit-exactly
FP8_MIN_SUBNORMAL = 2.0 ** -9

BENCH_N = 6144
CHUNK_SIZES = [2048, 4096, 8192, 16384, 32768]
TARGET_GBPS = 1600.0
COPY_WALL_GBPS = 1322.0
SENTINEL = 0xA5
GUARD = 4096                     # guard elements before/after outputs


# ------------------------------- reference -----------------------------------
def ref_quant(inp):
    """Returns (out_fp8 [m,n], scale_fp32 [m,G])."""
    m, n = inp.shape
    g = n // GROUP
    x = inp.float().view(m, g, GROUP)
    absmax = x.abs().amax(dim=2, keepdim=True).clamp_(min=FP8_MIN_ABSMAX)
    scale = (absmax / QMAX).view(m, g).contiguous()
    out = (x * (QMAX / absmax)).view(m, n).to(FP8)
    return out, scale


def cosine_sim(inp, out_fp8, scale_mg, chunk_rows=2048):
    m = inp.shape[0]
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros_like(dot)
    nb = torch.zeros_like(dot)
    for lo in range(0, m, chunk_rows):
        hi = min(lo + chunk_rows, m)
        ref = inp[lo:hi].double()
        deq = out_fp8[lo:hi].double() * scale_mg[lo:hi].double().repeat_interleave(GROUP, 1)
        dot += (deq * ref).sum()
        na += (deq * deq).sum()
        nb += (ref * ref).sum()
    if nb.item() == 0.0 and na.item() == 0.0:
        return 1.0
    return (dot / (na.sqrt() * nb.sqrt()).clamp_min(1e-300)).item()


# ------------------------- guarded output allocation -------------------------
class Guarded:
    """Contiguous view inside a larger sentinel-filled buffer to catch OOB writes."""

    def __init__(self, shape, dtype):
        numel = 1
        for s in shape:
            numel *= s
        self.dtype = dtype
        if dtype == FP8:
            self.raw = torch.full((numel + 2 * GUARD,), SENTINEL, dtype=torch.uint8, device=DEVICE)
            self.tensor = self.raw[GUARD:GUARD + numel].view(FP8).view(*shape)
        else:
            self.raw = torch.full((numel + 2 * GUARD,), float("nan"), dtype=dtype, device=DEVICE)
            self.tensor = self.raw[GUARD:GUARD + numel].view(*shape)
        self.numel = numel

    def guards_intact(self):
        head, tail = self.raw[:GUARD], self.raw[GUARD + self.numel:]
        if self.dtype == FP8:
            return bool((head == SENTINEL).all() and (tail == SENTINEL).all())
        return bool(head.isnan().all() and tail.isnan().all())


# --------------------------------- runner ------------------------------------
def run_op(out, scale, inp, trans_scale=False):
    torch.ops.sgl_kernel.per_token_cast_to_fp8(out, scale, inp, trans_scale)


def check_case(m, n, dtype=torch.bfloat16, trans_scale=False, kind="randn",
               seed=0, verbose=True):
    g = n // GROUP
    inp = make_input(m, n, dtype, kind, seed)

    out_g = Guarded((m, n), FP8)
    scale_shape = (g, m) if trans_scale else (m, g)
    scale_g = Guarded(scale_shape, torch.float32)

    run_op(out_g.tensor, scale_g.tensor, inp, trans_scale)
    torch.cuda.synchronize()

    out, scale = out_g.tensor, scale_g.tensor
    scale_mg = scale.t().contiguous() if trans_scale else scale
    out_ref, scale_ref = ref_quant(inp)

    errs = []
    if not out_g.guards_intact():
        errs.append("OOB write on out")
    if not scale_g.guards_intact():
        errs.append("OOB write on scale")
    if torch.isnan(scale_mg).any() or torch.isinf(scale_mg).any():
        errs.append("scale contains nan/inf (unwritten entries?)")
    if not torch.allclose(scale_mg, scale_ref, rtol=SCALE_RTOL, atol=0):
        d = (scale_mg - scale_ref).abs().max().item()
        errs.append(f"scale mismatch max_abs={d:.3e}")

    of, rf = out.float(), out_ref.float()
    if torch.isnan(of).any():
        errs.append("out contains nan")
    match = (out.view(torch.uint8) == out_ref.view(torch.uint8)).float().mean().item()
    if match < FP8_MATCH_RATIO:
        errs.append(f"fp8 bit-match ratio {match:.4f} < {FP8_MATCH_RATIO}")
    # Allow at most 1 fp8 ULP (relative 2^-3, or subnormal step)
    tol = torch.maximum(rf.abs() * 0.125, torch.full_like(rf, FP8_MIN_SUBNORMAL)) + 1e-6
    if ((of - rf).abs() > tol).any():
        errs.append(f"fp8 diff > 1ulp, max={(of - rf).abs().max().item():.4f}")
    if of.abs().max().item() > QMAX:
        errs.append("fp8 value exceeds 448")

    sim = cosine_sim(inp, out, scale_mg)
    if kind not in ("zeros", "tiny") and sim < COS_THRESHOLD:
        errs.append(f"cos_sim {sim:.6f} < {COS_THRESHOLD}")

    ok = not errs
    if verbose:
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] m={m:6d} n={n:5d} {str(dtype).split('.')[-1]:8s} "
              f"trans={int(trans_scale)} kind={kind:9s} cos={sim:.6f} match={match:.4f}"
              + ("" if ok else "  <- " + "; ".join(errs)))
    return ok


def make_input(m, n, dtype, kind, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(m, n, dtype=torch.float32, device=DEVICE, generator=gen)
    if kind == "randn":
        pass
    elif kind == "zeros":
        x.zero_()
    elif kind == "tiny":           # below absmax floor 1e-4
        x = x * 1e-6
    elif kind == "large":
        x = x * 1e4
    elif kind == "outlier":        # one big outlier per group
        x = x.view(m, n // GROUP, GROUP)
        x[:, :, 17] = 300.0
        x = x.view(m, n)
    elif kind == "negative":
        x = -x.abs()
    elif kind == "const":
        x.fill_(3.0)
    elif kind == "mixscale":       # per-group magnitude varies 1e-3 .. 1e3
        s = torch.logspace(-3, 3, n // GROUP, device=DEVICE).repeat_interleave(GROUP)
        x = x * s
    else:
        raise ValueError(kind)
    return x.to(dtype).contiguous()


# --------------------------------- tests -------------------------------------
PATH_SHAPES = [
    # (m, n)          path when trans_scale=False
    (1, 128),         # small
    (7, 384),         # small, G=3
    (256, 6144),      # small boundary (m==256)
    (257, 6144),      # VEC16 just above small
    (300, 1024),      # VEC4  (307200 elems)
    (333, 256),       # VEC4, odd m
    (600, 1024),      # VEC8  (614400 elems)
    (1000, 640),      # VEC8, tail block
    (2048, 6144),     # VEC16
    (4097, 6144),     # VEC16, odd m tail
]

TRANS_SHAPES = [
    (1, 128),         # scatter (m<32)
    (16, 6144),       # scatter (m<32, G=48)
    (100, 512),       # scatter (G=4<8)
    (31, 1024),       # scatter (m=31 boundary)
    (32, 1024),       # workspace (G=8, m=32 boundary)
    (64, 896),        # scatter (G=7 boundary)
    (1000, 1152),     # workspace, non-multiple-of-32 m/G
    (2048, 6144),     # workspace, VEC16
]

SPECIAL_KINDS = ["zeros", "tiny", "large", "outlier", "negative", "const", "mixscale"]


def test_paths():
    ok = True
    for dt in (torch.bfloat16, torch.float16):
        for m, n in PATH_SHAPES:
            ok &= check_case(m, n, dt, trans_scale=False)
    assert ok


def test_trans_scale():
    ok = True
    for dt in (torch.bfloat16, torch.float16):
        for m, n in TRANS_SHAPES:
            ok &= check_case(m, n, dt, trans_scale=True)
    assert ok


def test_special_values():
    ok = True
    for kind in SPECIAL_KINDS:
        for m, n, tr in [(64, 1024, False), (300, 1024, False),
                         (2048, 6144, False), (100, 512, True), (1000, 1152, True)]:
            ok &= check_case(m, n, torch.bfloat16, trans_scale=tr, kind=kind)
    assert ok


def test_zero_semantics():
    m, n = 64, 1024
    inp = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty(m, n, dtype=FP8, device=DEVICE)
    scale = torch.empty(m, n // GROUP, dtype=torch.float32, device=DEVICE)
    run_op(out, scale, inp)
    torch.cuda.synchronize()
    assert (out.float() == 0).all(), "zeros must quantize to 0"
    assert torch.allclose(scale, torch.full_like(scale, FP8_MIN_ABSMAX / QMAX), rtol=1e-6)
    print("  [OK  ] zero semantics (out==0, scale==1e-4/448)")


def test_absmax_maps_to_qmax():
    m, n = 128, 2048
    inp = make_input(m, n, torch.bfloat16, "randn", 7)
    out = torch.empty(m, n, dtype=FP8, device=DEVICE)
    scale = torch.empty(m, n // GROUP, dtype=torch.float32, device=DEVICE)
    run_op(out, scale, inp)
    torch.cuda.synchronize()
    gmax = out.float().abs().view(m, -1, GROUP).amax(-1)
    assert (gmax == QMAX).all(), f"group absmax should map to 448, min got {gmax.min().item()}"
    print("  [OK  ] per-group absmax maps to +-448")


def test_determinism():
    for m, n, tr in [(2048, 6144, False), (1000, 1152, True), (16, 6144, True)]:
        inp = make_input(m, n, torch.bfloat16, "randn", 3)
        g = n // GROUP
        res = []
        for _ in range(3):
            out = torch.empty(m, n, dtype=FP8, device=DEVICE)
            sc = torch.empty((g, m) if tr else (m, g), dtype=torch.float32, device=DEVICE)
            run_op(out, sc, inp, tr)
            res.append((out.view(torch.uint8).clone(), sc.clone()))
        torch.cuda.synchronize()
        for o, s in res[1:]:
            assert torch.equal(o, res[0][0]) and torch.equal(s, res[0][1])
    print("  [OK  ] determinism")


def test_trans_consistency():
    """trans_scale=True scale must equal transpose of trans_scale=False scale."""
    for m, n in TRANS_SHAPES:
        inp = make_input(m, n, torch.bfloat16, "randn", 11)
        g = n // GROUP
        o1 = torch.empty(m, n, dtype=FP8, device=DEVICE)
        o2 = torch.empty_like(o1)
        s1 = torch.empty(m, g, dtype=torch.float32, device=DEVICE)
        s2 = torch.empty(g, m, dtype=torch.float32, device=DEVICE)
        run_op(o1, s1, inp, False)
        run_op(o2, s2, inp, True)
        torch.cuda.synchronize()
        assert torch.equal(o1.view(torch.uint8), o2.view(torch.uint8)), (m, n)
        assert torch.equal(s1.t(), s2), (m, n)
    print("  [OK  ] trans_scale True/False consistency")


def _expect_error(fn, name):
    try:
        fn()
        torch.cuda.synchronize()
    except (RuntimeError, ValueError, TypeError):
        print(f"  [OK  ] rejects {name}")
        return True
    print(f"  [FAIL] did not reject {name}")
    return False


def test_invalid_args():
    m = 16
    ok = True
    # n not multiple of 128
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 200, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 200, dtype=torch.bfloat16, device=DEVICE)), "n % 128 != 0")
    # fp32 input
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 256, dtype=torch.float32, device=DEVICE)), "fp32 input")
    # wrong out dtype
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=torch.bfloat16, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 256, dtype=torch.bfloat16, device=DEVICE)), "non-fp8 out")
    # non-contiguous input
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(256, m, dtype=torch.bfloat16, device=DEVICE).t()), "non-contiguous input")
    assert ok


# ------------------------------- benchmark -----------------------------------
def bench_burst(fn, warm=0.5, run=0.6):
    torch.cuda.synchronize()
    t0 = time.time(); n = 0
    while time.time() - t0 < warm:
        fn(); n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(50, n)
    best_ms = float("inf"); t0 = time.time()
    while time.time() - t0 < run:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def effective_bytes(m, n):
    return m * n * 2 + m * n + m * (n // GROUP) * 4


def bench(trans_scale):
    n, g = BENCH_N, BENCH_N // GROUP
    peak, all_ok = 0.0, True
    for m in CHUNK_SIZES:
        inp = make_input(m, n, torch.bfloat16, "randn", 1234)
        out = torch.empty(m, n, dtype=FP8, device=DEVICE)
        sc = torch.empty((g, m) if trans_scale else (m, g), dtype=torch.float32, device=DEVICE)
        run_op(out, sc, inp, trans_scale)
        torch.cuda.synchronize()
        sim = cosine_sim(inp, out, sc.t() if trans_scale else sc)
        ok = sim >= COS_THRESHOLD
        ms = bench_burst(lambda: run_op(out, sc, inp, trans_scale))
        gbps = effective_bytes(m, n) / (ms * 1e-3) / 1e9
        peak = max(peak, gbps); all_ok &= ok
        print(f"  [bench trans={int(trans_scale)}] T={m:6d} G={g}  cos={sim:.6f} "
              f"{'OK' if ok else 'FAIL'}  {ms:8.4f} ms  {gbps:8.1f} GB/s")
        del inp, out, sc
        torch.cuda.empty_cache()
    print(f"  peak(trans={int(trans_scale)}) = {peak:.1f} GB/s  "
          f"(target {TARGET_GBPS:.0f}: {'REACHED' if peak >= TARGET_GBPS else 'NOT reached'}; "
          f"copy wall ~{COPY_WALL_GBPS:.0f})")
    return all_ok


# --------------------------------- main --------------------------------------
TESTS = [test_paths, test_trans_scale, test_special_values, test_zero_semantics,
         test_absmax_maps_to_qmax, test_determinism, test_trans_consistency,
         test_invalid_args]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} "
          f"device_count={torch.cuda.device_count()}")
    print("=" * 100)

    failed = []
    if not args.bench_only:
        for t in TESTS:
            print(f"[{t.__name__}]")
            try:
                t()
            except AssertionError as e:
                failed.append(t.__name__)
                print(f"  -> FAILED {e}")
    if not args.no_bench:
        print("[benchmark]")
        for tr in (False, True):
            if not bench(tr):
                failed.append(f"bench_accuracy(trans={tr})")

    print("=" * 100)
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()