"""Unit test + bandwidth benchmark for mcoplib top_k_per_row_decode /
top_k_per_row_prefill (vllm sampler.cu) on MetaX C500/C600U.

Ops (indices output is pre-allocated and mutated in place):
  top_k_per_row_decode(logits, next_n, seq_lens, indices,
                       num_rows, stride0, stride1, top_k)
    logits [B*next_n, L] fp32; seq_lens 1D [B] (shared per batch) or 2D [B, next_n].
    Row r -> b=r//next_n, j=r%next_n; row_end = seq_lens[b] (2D: seq_lens[b, j]) or
             seq_lens[b] - next_n + j + 1 (1D), clamped to [0, L].
    Top-`top_k` column indices of logits[r, :row_end] are written unordered to
    indices[r, :]; short rows (row_end <= top_k) get [0..row_end) then -1 padding.
  top_k_per_row_prefill(logits, row_starts, row_ends, indices,
                        num_rows, stride0, stride1, top_k)
    Ragged rows [row_starts[i], row_ends[i]) inside logits[i, :L]; output indices
    are RELATIVE to row_starts[i]; short rows padded with -1.

Accuracy contract (STRICT, rtol=0 atol=0): for every row the multiset of gathered
values must exactly equal torch.topk values (ties broken on identical float bits,
so any index choice among bit-equal values is valid), indices distinct and in
range; short rows must equal arange(row_len) + (-1) tail exactly.

Model shape coverage (DeepSeek-V4-FLASH / V4-Pro / V4.1-PRO/FLASH indexer):
  decode  : next_n in {1,2,3,4,8}, batch in {1..128}, context len 2K..256K,
            top_k mainly 512 / 1024 (both mandatory) + small K generality sweep.
  prefill : ragged rows 1..512, row len 2K..256K, same top_k set.
  Edges   : dispatch boundaries (13056 smem cap, 12288/200000 legacy), short rows,
            ragged seq_lens, 2D seq_lens, non-contiguous logits (stride-2), tie /
            all-equal / zeros / +-inf patterns, CUDA-graph capture & replay.

Bandwidth (GB/s) = (rows*L*4 + rows*seq_bytes + rows*top_k*4) / t
  [decode seq_bytes=4 (seq_lens), prefill seq_bytes=8 (row_starts+row_ends)]

Run:
  mx-smi            # pick an idle GPU
  export CUDA_VISIBLE_DEVICES=<idle>
  python -u unit_test/test_top_k_per_row_prefill_decode.py
"""

import os
import torch

import mcoplib._C  # noqa: F401  (registers torch.ops._C)

DEVICE = "cuda"
DTYPE = torch.float32
TARGET_GBPS = 1300.0          # single-die C500 target
WARMUP = 10
REPS = 50

SEQ_BYTES_DECODE = 4          # seq_lens read per row
SEQ_BYTES_PREFILL = 8         # row_starts + row_ends read per row

# ---------------------------------------------------------------- case tables

# DSV4-FLASH/Pro/V4.1 decode sweep: context length axis doubles as the model
# family axis (FLASH <= 32K, Pro 48K-96K, V4.1-PRO long ctx 128K-256K).
MAIN_L = (2048, 4096, 6144, 8192, 12288, 13056, 16384, 20480,
          32768, 49152, 65536, 98304, 131072, 196608, 262144)
DECODE_MAIN_BATCH = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 96, 128)
DECODE_MAIN_NN = (1, 2, 3, 4, 8)
PREFILL_MAIN_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
MAIN_TOPK = (512, 1024)

BOUNDARY_L = ()
SMALL_TOPK = ()
SMALL_TOPK_L = ()
SHORT_L = ()

PATTERNS = ("randn", "ties_at_cut", "all_equal", "many_zeros",
            "pos_inf_dup", "neg_inf_mix")
PATTERN_L = (4096, 65536, 262144)

MAX_BYTES = 1 << 30           # 1 GB logits guard for the biggest combos


def _gen(seed, shape):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DTYPE, device=DEVICE)


def make_logits(rows, L, seed, pattern="randn", noncontig=False):
    if pattern == "randn":
        base = _gen(seed, (rows, L))
    elif pattern == "ties_at_cut":
        base = _gen(seed, (rows, L)) * 0.1
        base[:, : min(2 * 1024, L)] = 1.0        # cut falls inside equal block
    elif pattern == "all_equal":
        base = torch.full((rows, L), 0.25, dtype=DTYPE, device=DEVICE)
    elif pattern == "many_zeros":
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        keep = torch.rand((rows, L), generator=g, device=DEVICE) < 0.4
        base = _gen(seed + 1, (rows, L)) * keep  # ~60% exact +-0.0
    elif pattern == "pos_inf_dup":
        base = _gen(seed, (rows, L)) * 0.1
        base[:, : min(1024 + 512, L)] = float("inf")
    elif pattern == "neg_inf_mix":
        base = _gen(seed, (rows, L))
        base[:, ::7] = float("-inf")
    else:
        raise ValueError(pattern)
    if not noncontig:
        return base
    if pattern == "randn":
        return _gen(seed + 2, (rows, L * 2))[:, ::2]
    wide = torch.empty((rows, L * 2), dtype=DTYPE, device=DEVICE)
    wide[:, ::2] = base
    wide[:, 1::2] = -7.0                        # never read (stride-2 view)
    return wide[:, ::2]


def expected_row_ends(L, next_n, seq_lens):
    """[rows] row_end for decode, matching kernel semantics (clamped to [0, L])."""
    if seq_lens.dim() == 2:                      # [B, next_n] flat = b*nn + j
        re = seq_lens.reshape(-1).to(torch.int64)
    else:                                        # [B] shared
        rows = seq_lens.numel() * next_n
        ar = torch.arange(rows, device=DEVICE)
        re = (seq_lens[ar // next_n].to(torch.int64)
              - next_n + (ar % next_n) + 1)
    return re.clamp_(0, L)


def check_case(indices, logits, row_ends, row_starts, topk, L):
    """Returns (n_bad_rows, n_mismatch_rows). Strict equality contract."""
    rows = logits.size(0)
    if row_starts is None:
        row_starts = torch.zeros(rows, dtype=torch.int64, device=DEVICE)
    else:
        row_starts = row_starts.to(torch.int64)
    row_ends = row_ends.to(torch.int64)
    row_lens = (row_ends - row_starts).clamp_(min=0)
    arange_L = torch.arange(L, device=DEVICE)[None, :]
    mask = (arange_L >= row_starts[:, None]) & (arange_L < row_ends[:, None])
    masked = logits.masked_fill(~mask, float("-inf"))
    k_eff = min(topk, L)
    exp_vals = torch.topk(masked, k_eff, dim=1).values
    if k_eff < topk:
        exp_vals = torch.cat(
            [exp_vals, torch.full((rows, topk - k_eff), float("-inf"),
                                  dtype=DTYPE, device=DEVICE)], dim=1)

    valid = indices >= 0
    abs_idx = row_starts[:, None] + indices.clamp(min=0).to(torch.int64)
    gathered = logits.gather(1, abs_idx)
    gathered = gathered.masked_fill(~valid, float("-inf"))
    got_vals = gathered.sort(dim=1, descending=True).values
    mismatch = ~(got_vals == exp_vals).all(dim=1)          # exact bit compare

    bad = mismatch.clone()
    short = row_lens < topk
    if short.any():
        # short rows: deterministic contract arange(row_len) + (-1) tail
        arange_K = torch.arange(topk, device=DEVICE)[None, :]
        exp_short = torch.where(arange_K < row_lens[:, None], arange_K,
                                torch.full_like(arange_K, -1))
        bad[short] = ~(indices[short] == exp_short[short]).all(dim=1)
    long_rows = ~short
    if long_rows.any():
        idx = indices[long_rows]
        bad_long = ~(idx >= 0).all(dim=1)
        srt = idx.sort(dim=1).values
        bad_long |= ~(srt[:, 1:] > srt[:, :-1]).all(dim=1)          # distinct
        bad_long |= ~(idx < row_lens[long_rows, None]).all(dim=1)   # in range
        bad[long_rows] |= bad_long
    return int(bad.sum().item()), int(mismatch.sum().item())


def bench(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


# ---------------------------------------------------------------- decode cases

def run_decode(t):
    torch.ops._C.top_k_per_row_decode(
        t["logits"], t["next_n"], t["seq_lens"], t["indices"],
        t["rows"], t["logits"].stride(0), t["logits"].stride(1), t["topk"])


def decode_case(batch, next_n, L, topk, seed, pattern="randn",
                two_d=False, noncontig=False, seq_mode="full", tag=""):
    rows = batch * next_n
    logits = make_logits(rows, L, seed, pattern, noncontig)
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    if seq_mode == "full":
        sl = torch.full((batch,), L, dtype=torch.int32, device=DEVICE)
    elif seq_mode == "ragged":
        lo = max(topk + next_n, L // 8)
        sl = torch.randint(lo, L + 1, (batch,), generator=g,
                           dtype=torch.int32, device=DEVICE)
    elif seq_mode == "tiny":
        sl = torch.randint(0, min(next_n + 2, L + 1), (batch,), generator=g,
                           dtype=torch.int32, device=DEVICE)
    else:
        raise ValueError(seq_mode)
    if two_d:
        sl2 = sl.reshape(batch, 1).repeat(1, next_n).contiguous()
        jitter = torch.randint(0, next_n + 1, (batch, next_n), generator=g,
                               device=DEVICE, dtype=torch.int32)
        sl = (sl2 - jitter).clamp_(min=0)
    indices = torch.full((rows, topk), -1, dtype=torch.int32, device=DEVICE)
    row_ends = expected_row_ends(L, next_n, sl)
    return dict(op="decode", tag=tag or pattern, logits=logits, next_n=next_n,
                seq_lens=sl, indices=indices, rows=rows, L=L, topk=topk,
                row_ends=row_ends, row_starts=None)


# --------------------------------------------------------------- prefill cases

def run_prefill(t):
    torch.ops._C.top_k_per_row_prefill(
        t["logits"], t["row_starts"], t["row_ends"], t["indices"],
        t["rows"], t["logits"].stride(0), t["logits"].stride(1), t["topk"])


def prefill_case(rows, L, topk, seed, pattern="randn", noncontig=False,
                 ragged="uniform", tag=""):
    logits = make_logits(rows, L, seed, pattern, noncontig)
    g = torch.Generator(device=DEVICE).manual_seed(seed + 2)
    if ragged == "uniform":
        rs = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
        re = torch.full((rows,), L, dtype=torch.int32, device=DEVICE)
    elif ragged == "ragged":
        rs = torch.randint(0, max(1, L // 8), (rows,), generator=g,
                           dtype=torch.int32, device=DEVICE)
        lens = torch.randint(0, L + 1, (rows,), generator=g,
                             dtype=torch.int32, device=DEVICE)
        re = (rs + lens).clamp_(max=L)
    elif ragged == "mixed":
        rs = torch.randint(0, max(1, L // 4), (rows,), generator=g,
                           dtype=torch.int32, device=DEVICE)
        lens = torch.randint(1, min(2 * topk, L) + 1, (rows,), generator=g,
                             dtype=torch.int32, device=DEVICE)
        re = (rs + lens).clamp_(max=L)
    else:
        raise ValueError(ragged)
    indices = torch.full((rows, topk), -1, dtype=torch.int32, device=DEVICE)
    return dict(op="prefill", tag=tag or f"{pattern}/{ragged}", logits=logits,
                row_starts=rs, row_ends=re, indices=indices, rows=rows, L=L,
                topk=topk)


# ---------------------------------------------------------------- graph cases

def graph_case(t, run_fn, n_replay=3):
    """Capture the op, replay, and verify strict equality after each replay.
    Replay 2+ mutates logits (and seq_lens/row_ends) in place so data-dependent
    row lengths flow through the captured graph."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            run_fn(t)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fn(t)
    bad_total = mism_total = 0
    L = t["L"]
    for it in range(n_replay):
        if it > 0:
            t["logits"].normal_()
            if t["op"] == "decode":
                lo = max(t["topk"] + t["next_n"], L // 4)
                t["seq_lens"].random_(lo, L + 1)
                t["row_ends"] = expected_row_ends(L, t["next_n"], t["seq_lens"])
            else:
                t["row_starts"].random_(0, max(1, L // 4))
                t["row_ends"].random_(max(1, L // 2), L + 1)
        g.replay()
        torch.cuda.synchronize()
        b, m = check_case(t["indices"], t["logits"], t["row_ends"],
                          t["row_starts"], t["topk"], L)
        bad_total += b
        mism_total += m
    return bad_total, mism_total


# ------------------------------------------------------------------ main loop

def execute(t, run_fn, stats):
    if t["logits"].numel() * 4 > MAX_BYTES:
        stats["skipped"] += 1
        return
    run_fn(t)
    torch.cuda.synchronize()
    bad, mism = check_case(t["indices"], t["logits"], t["row_ends"],
                           t["row_starts"], t["topk"], t["L"])
    ms = bench(lambda: run_fn(t))
    # Effective bytes: only the columns actually read per row matter (ragged /
    # short rows read far less than L). row_starts is None for decode (rowStart=0).
    rs = torch.zeros(t["rows"], dtype=torch.int32, device=DEVICE) \
        if t["row_starts"] is None else t["row_starts"]
    row_lens = (t["row_ends"].to(torch.int64)
                - rs.to(torch.int64)).clamp_(min=0)
    read_elem = int(row_lens.sum().item())
    seq_b = SEQ_BYTES_DECODE if t["op"] == "decode" else SEQ_BYTES_PREFILL
    gbps = (read_elem * 4 + t["rows"] * seq_b
            + t["rows"] * t["topk"] * 4) / (ms * 1e-3) / 1e9
    ok = "OK" if bad == 0 else "FAIL"
    if bad:
        stats["fail"] += 1
    else:
        stats["pass"] += 1
    stats["peak"] = max(stats["peak"], gbps)
    nn = t["next_n"] if t["op"] == "decode" else "-"
    print(f"[{t['op']:>7}] L={t['L']:6d} rows={t['rows']:5d} nn={nn} "
          f"K={t['topk']:4d} tag={t['tag']:<14.14} bad={bad:3d} {ok:4s} "
          f"{ms:9.4f} ms {gbps:8.1f} GB/s")


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  dtype=float32/int32  topk={MAIN_TOPK} (+{SMALL_TOPK}) "
          f"warmup={WARMUP} reps={REPS}")
    print("=" * 118)

    cases = []
    # ---- decode main sweep (DSV4-FLASH/Pro/V4.1) ----
    for topk in MAIN_TOPK:
        for next_n in DECODE_MAIN_NN:
            for batch in DECODE_MAIN_BATCH:
                for L in MAIN_L:
                    cases.append(("decode", dict(batch=batch, next_n=next_n,
                                                 L=L, topk=topk)))
    # ---- decode boundaries / small K / short rows / ragged / 2D / noncontig ----
    for L in BOUNDARY_L:
        for topk in MAIN_TOPK:
            for batch in (2, 32):
                cases.append(("decode", dict(batch=batch, next_n=1, L=L,
                                             topk=topk, tag="boundary")))
    for topk in SMALL_TOPK:
        for L in SMALL_TOPK_L:
            for batch in (1, 8):
                for nn in (1, 2):
                    cases.append(("decode", dict(batch=batch, next_n=nn, L=L,
                                                 topk=topk, tag="smallK")))
    for L in SHORT_L:
        for topk in MAIN_TOPK:
            for batch in (1, 4):
                for nn in (1, 4):
                    cases.append(("decode", dict(batch=batch, next_n=nn, L=L,
                                                 topk=topk, tag="short")))
    for seq_mode in ("ragged", "tiny"):
        for topk in MAIN_TOPK:
            for batch in (2, 8, 32):
                for nn in (1, 4):
                    cases.append(("decode", dict(batch=batch, next_n=nn, L=131072,
                                                 topk=topk, seq_mode=seq_mode,
                                                 tag=f"seq-{seq_mode}")))
    for topk in MAIN_TOPK:
        for nn in (2, 4, 8):
            for batch in (2, 4):
                for L in (8192, 131072):
                    cases.append(("decode", dict(batch=batch, next_n=nn, L=L,
                                                 topk=topk, two_d=True,
                                                 tag="seq2d")))
        for L in (4096, 65536):
            for batch in (2, 16):
                for nn in (1, 2):
                    cases.append(("decode", dict(batch=batch, next_n=nn, L=L,
                                                 topk=topk, noncontig=True,
                                                 tag="stride2")))
        for pat in PATTERNS:
            for L in PATTERN_L:
                for batch in (1, 8):
                    cases.append(("decode", dict(batch=batch, next_n=1, L=L,
                                                 topk=topk, pattern=pat)))
    # ---- prefill main sweep + edges ----
    for topk in MAIN_TOPK:
        for rows in PREFILL_MAIN_ROWS:
            for L in MAIN_L:
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk)))
    for L in BOUNDARY_L:
        for topk in MAIN_TOPK:
            for rows in (2, 32):
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                              tag="boundary")))
    for topk in SMALL_TOPK:
        for L in SMALL_TOPK_L:
            for rows in (1, 8):
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                              tag="smallK")))
    for L in SHORT_L:
        for topk in MAIN_TOPK:
            for rows in (2, 8):
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                              tag="short")))
    for rag in ("ragged", "mixed"):
        for topk in MAIN_TOPK:
            for (rows, L) in ((2, 4096), (8, 13056), (16, 65536), (32, 131072),
                              (64, 4096), (128, 262144)):
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                              ragged=rag, tag=f"rag-{rag}")))
    for topk in MAIN_TOPK:
        for L in (4096, 65536):
            for rows in (2, 16):
                cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                              noncontig=True, tag="stride2")))
        for pat in PATTERNS:
            for L in PATTERN_L:
                for rows in (1, 8):
                    cases.append(("prefill", dict(rows=rows, L=L, topk=topk,
                                                  pattern=pat)))

    total = len(cases)
    print(f"total cases: {total} (target >= 2000)")
    assert total >= 2000, "case matrix below 2000"

    stats = {"pass": 0, "fail": 0, "skipped": 0, "peak": 0.0}
    seed = 1234
    for i, (op, kw) in enumerate(cases):
        seed += 1
        if op == "decode":
            t = decode_case(seed=seed, **kw)
            execute(t, run_decode, stats)
        else:
            t = prefill_case(seed=seed, **kw)
            execute(t, run_prefill, stats)
        if (i + 1) % 400 == 0:
            torch.cuda.empty_cache()

    # ---- CUDA graph section ----
    print("---- CUDA graph capture/replay " + "-" * 60)
    graph_specs = [
        ("decode", dict(batch=2, next_n=1, L=4096, topk=1024, tag="graph")),
        ("decode", dict(batch=4, next_n=2, L=13056, topk=1024, tag="graph")),
        ("decode", dict(batch=1, next_n=1, L=131072, topk=512, tag="graph")),
        ("decode", dict(batch=2, next_n=1, L=262144, topk=1024, tag="graph")),
        ("decode", dict(batch=2, next_n=4, L=65536, topk=1024, two_d=True,
                        tag="graph2d")),
        ("prefill", dict(rows=4, L=4096, topk=1024, tag="graph")),
        ("prefill", dict(rows=8, L=13056, topk=512, tag="graph")),
        ("prefill", dict(rows=2, L=262144, topk=1024, tag="graph")),
        ("prefill", dict(rows=16, L=65536, topk=1024, ragged="ragged",
                         tag="graphRagged")),
    ]
    for op, kw in graph_specs:
        seed += 1
        if op == "decode":
            t = decode_case(seed=seed, **kw)
            bad, _ = graph_case(t, run_decode)
        else:
            t = prefill_case(seed=seed, **kw)
            bad, _ = graph_case(t, run_prefill)
        ok = "OK" if bad == 0 else "FAIL"
        stats["pass"] += 1 if bad == 0 else 0
        stats["fail"] += 0 if bad == 0 else 1
        print(f"[{op:>7}] graph replay x3  L={t['L']:6d} rows={t['rows']:5d} "
              f"K={t['topk']:4d} bad={bad:3d} {ok}")

    print("=" * 118)
    acc = "ALL PASS" if stats["fail"] == 0 else "FAILURES PRESENT"
    print(f"Accuracy: {acc}  (passed {stats['pass']}, failed {stats['fail']}, "
          f"skipped {stats['skipped']})")
    print(f"Peak bandwidth: {stats['peak']:.1f} GB/s  "
          f"({'REACHED' if stats['peak'] >= TARGET_GBPS else 'below'} "
          f"single-die target {TARGET_GBPS:.0f} GB/s)")
    assert stats["fail"] == 0, f"accuracy check failed ({stats['fail']} cases)"


if __name__ == "__main__":
    main()
