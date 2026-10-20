"""Unit test + bandwidth benchmark for mcoplib.op.fused_rmsnorm_rope_quant_reshape_and_cache
on MetaX C600-U.

Op (17-arg, in-place):
  fused_rmsnorm_rope_quant_reshape_and_cache(
      packed_qkv,            bf16  [T, (Hq + 2*Hkv) * 128]      (Q segment updated in place)
      q_norm_weight,         fp32  [128]
      k_norm_weight,         fp32  [128]
      cos, sin,              fp32  [max_pos, rope_dim]          (NeoX layout, halves equal)
      q_lens,                int32 [B]                          (interface only)
      cache_lens,            int32 [B]                          (history length per seq)
      accum_q_lens,          int32 [B+1]                        (prefix sum of q_lens)
      k_cache,               [nb, Hkv, 128/8, block_size, 8]
      v_cache,               [nb, Hkv, 128, block_size]
      slot_mapping,          int64 [T]                          (<0 => skip)
      k_scale, v_scale,      fp32  [1]                          (required for fp8 / int8)
      kv_cache_dtype,        "bf16" | "bfloat16" | "fp8" | "fp8_e4m3" | "float8" | "int8"
      rope_offset, block_size, eps)

Semantics (per token t, pos = cache_lens[b] + (t - accum_q_lens[b])):
  Q heads : q = rope_neox(rmsnorm(q, q_w), pos)          -> bf16, written back into packed_qkv
  K heads : k = rope_neox(rmsnorm(k, k_w), pos) / k_scale -> quant -> k_cache[slot]
  V heads : v / v_scale                                   -> quant -> v_cache[slot]
  quant   : fp8 = clamp(+-448) e4m3 RNE ; int8 = clamp(round_half_even, +-127) ; bf16 = RNE
  K/V segments of packed_qkv are NOT modified.

Kernel paths:
  head-major QK kernel   : num_tokens < 4096 (all decode cases)
  token-major QK kernel  : num_tokens >= 4096 and rope_dim == 128 (with offset aligned to 32)
  V kernel VST=8         : fp8 and num_tokens < 256 and num_tokens % 16 == 8, otherwise VST=16
  BATCH1 specialization  : batch_size == 1

Suites:
  full : bf16/fp8/int8 cache, rope_dim 64/128, decode sweep + prefill (covers token-major/batch1)
  json : cases from JSON_CASES (or --json-file), same schema as the op-config json
  all  : full + json (default)

Run:
  export CUDA_VISIBLE_DEVICES=0
  python unit_test/test_op_fused_rmsnorm_rope_quant_reshape_and_cache.py \
      [--suite all|full|json] [--json-file cfg.json] [--baseline]
"""

import argparse
import itertools
import json
import os
import torch

import mcoplib.op as ops

DEVICE = "cuda"
OP_NAME = "fused_rmsnorm_rope_quant_reshape_and_cache"

EPS = 1e-6
X = 8                      # k_cache innermost dim (16B of bf16)
K_SCALE = 0.05
V_SCALE = 0.02
TARGET_GBPS = 1300.0       # single-die gate
DUAL_DIE_GBPS = 3480.0     # dual-die datasheet figure

FLOAT8 = torch.float8_e4m3fn

SRC_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16}

# cache dtype -> (op string, torch dtype, elem bytes, quant step for max-err, cos thr K/V)
CACHE_DTYPES = {
    "bf16":   ("bf16",   torch.bfloat16, 2, None, 0.99999),
    "float8": ("float8", FLOAT8,         1, None, 0.999),
    "int8":   ("int8",   torch.int8,     1, 1.0,  0.9999),
}
CACHE_ALIASES = {"bfloat16": "bf16", "fp8": "float8", "fp8_e4m3": "float8",
                 "float8_e4m3fn": "float8"}
Q_COS_THR = 0.99999

# ---------------- suite "full" ----------------
FULL_CASES = {
    OP_NAME: [
        {   # decode sweep: head-major kernel, both V VST paths
            "arg_type": "llm", "dtype": "bfloat16",
            "cache_dtype": ["bf16", "float8", "int8"],
            "attn_mode": "decode", "rope_dim": [64, 128], "rope_offset": 0,
            "q_head_num.kv_head_num.head_dim": [[8, 1, 128], [32, 8, 128], [64, 4, 128]],
            "block_size": 64,
            "q_len": [1, 4],
            "batch_size": [1, 8, 16, 24, 40, 64, 128, 208, 256],
            "cache_len": [10240],
        },
        {   # prefill: num_tokens >= 4096 -> token-major kernel, batch1 specialization
            "arg_type": "llm", "dtype": "bfloat16",
            "cache_dtype": ["bf16", "float8", "int8"],
            "attn_mode": "prefill", "rope_dim": 128, "rope_offset": 0,
            "q_head_num.kv_head_num.head_dim": [[8, 1, 128], [64, 4, 128]],
            "block_size": 64,
            "q_len": [1024, 4096],
            "batch_size": [1, 4],
            "cache_len": [0, 2048],
        },
    ]
}

# ---------------- suite "json" (from op-config json) ----------------
JSON_CASES = {
    OP_NAME: [
        {
            "arg_type": "llm",
            "dtype": "bfloat16",
            "cache_dtype": ["float8"],
            "attn_mode": "decode",
            "rope_dim": 128,
            "rope_offset": 0,
            "q_head_num.kv_head_num.head_dim": [[8, 1, 128]],
            "block_size": 64,
            "q_len": [1, 4],
            "batch_size": [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
                           128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208],
            "cache_len": [10240],
        }
    ]
}


def _as_list(v):
    return v if isinstance(v, list) else [v]


def expand_cases(cfg):
    """Expand op-config json into groups: (cfg_dict, [(q_len, batch, cache_len), ...])."""
    groups = []
    for entry in cfg.get(OP_NAME, []):
        if entry.get("arg_type", "llm") != "llm":
            continue
        src = entry.get("dtype", "bfloat16")
        assert src in SRC_DTYPES, f"unsupported dtype {src} (kernel is bf16-only)"
        shapes = list(itertools.product(_as_list(entry["q_len"]),
                                        _as_list(entry["batch_size"]),
                                        _as_list(entry["cache_len"])))
        for cache_name in _as_list(entry["cache_dtype"]):
            cache_name = CACHE_ALIASES.get(cache_name, cache_name)
            assert cache_name in CACHE_DTYPES, f"unsupported cache_dtype {cache_name}"
            for rope_dim in _as_list(entry["rope_dim"]):
                rope_off = int(entry.get("rope_offset", 0))
                for hq, hkv, hd in entry["q_head_num.kv_head_num.head_dim"]:
                    assert hd == 128, "fused path requires head_dim == 128"
                    assert rope_dim in (64, 128) and rope_off + rope_dim <= hd
                    assert rope_off % 8 == 0 and (rope_off // 8) % (rope_dim // 8) == 0
                    c = dict(src=src, cache=cache_name, mode=entry.get("attn_mode", ""),
                             rope_dim=rope_dim, rope_offset=rope_off, hq=hq, hkv=hkv,
                             head_dim=hd, block_size=int(entry["block_size"]))
                    groups.append((c, shapes))
    return groups


# ---------------- inputs ----------------
def make_inputs(batch, q_len, cache_len, c, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hq, hkv, D, bs, rd = c["hq"], c["hkv"], c["head_dim"], c["block_size"], c["rope_dim"]
    T = batch * q_len
    cache_dtype = CACHE_DTYPES[c["cache"]][1]

    packed_qkv = torch.randn(T, (hq + 2 * hkv) * D, device=DEVICE, generator=g).to(torch.bfloat16)
    q_w = (1.0 + 0.1 * torch.randn(D, device=DEVICE, generator=g)).contiguous()
    k_w = (1.0 + 0.1 * torch.randn(D, device=DEVICE, generator=g)).contiguous()

    max_pos = cache_len + q_len
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rd, 2, device=DEVICE).float() / rd))
    freqs = torch.outer(torch.arange(max_pos, device=DEVICE).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()

    q_lens = torch.full((batch,), q_len, dtype=torch.int32, device=DEVICE)
    cache_lens = torch.full((batch,), cache_len, dtype=torch.int32, device=DEVICE)
    accum = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    accum[1:] = torch.cumsum(q_lens, 0)

    # positions -> per-seq logical blocks -> random distinct physical blocks
    pos = (cache_len + torch.arange(q_len, device=DEVICE).repeat(batch)).long()
    seq = torch.arange(batch, device=DEVICE).repeat_interleave(q_len)
    blocks_per_seq = (max_pos + bs - 1) // bs
    logical = seq * blocks_per_seq + pos // bs
    uniq, inv = torch.unique(logical, return_inverse=True)
    num_blocks = uniq.numel() + 4
    phys = torch.randperm(num_blocks, device=DEVICE, generator=g)[:uniq.numel()]
    slot_mapping = (phys[inv] * bs + pos % bs).long().contiguous()

    k_cache = torch.zeros(num_blocks, hkv, D // X, bs, X, device=DEVICE, dtype=cache_dtype)
    v_cache = torch.zeros(num_blocks, hkv, D, bs, device=DEVICE, dtype=cache_dtype)
    k_scale = torch.tensor([K_SCALE], device=DEVICE, dtype=torch.float32)
    v_scale = torch.tensor([V_SCALE], device=DEVICE, dtype=torch.float32)

    return dict(packed_qkv=packed_qkv, q_w=q_w, k_w=k_w, cos=cos, sin=sin,
                q_lens=q_lens, cache_lens=cache_lens, accum=accum,
                k_cache=k_cache, v_cache=v_cache, slot_mapping=slot_mapping,
                k_scale=k_scale, v_scale=v_scale, pos=pos, T=T)


def run_op(t, c):
    ops.fused_rmsnorm_rope_quant_reshape_and_cache(
        t["packed_qkv"], t["q_w"], t["k_w"], t["cos"], t["sin"],
        t["q_lens"], t["cache_lens"], t["accum"],
        t["k_cache"], t["v_cache"], t["slot_mapping"],
        t["k_scale"], t["v_scale"], CACHE_DTYPES[c["cache"]][0],
        c["rope_offset"], c["block_size"], EPS)


# ---------------- reference ----------------
def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope_neox(x, cos, sin, off, rd):
    half = rd // 2
    x1, x2 = x[..., off:off + half], x[..., off + half:off + rd]
    cc, ss = cos[:, None, :half], sin[:, None, :half]
    out = x.clone()
    out[..., off:off + half] = x1 * cc - x2 * ss
    out[..., off + half:off + rd] = x2 * cc + x1 * ss
    return out


def quant(x, cache):
    if cache == "bf16":
        return x.to(torch.bfloat16)
    if cache == "int8":
        return torch.round(x).clamp(-127, 127).to(torch.int8)
    return x.clamp(-448.0, 448.0).to(FLOAT8)


def reference(t, c):
    hq, hkv, D = c["hq"], c["hkv"], c["head_dim"]
    qkv = t["packed_qkv"].float().view(t["T"], hq + 2 * hkv, D)
    q, k, v = qkv[:, :hq], qkv[:, hq:hq + hkv], qkv[:, hq + hkv:]
    cos, sin = t["cos"][t["pos"]], t["sin"][t["pos"]]
    q = rope_neox(rmsnorm(q, t["q_w"]), cos, sin, c["rope_offset"], c["rope_dim"])
    k = rope_neox(rmsnorm(k, t["k_w"]), cos, sin, c["rope_offset"], c["rope_dim"])
    ks = 1.0 if c["cache"] == "bf16" else K_SCALE
    vs = 1.0 if c["cache"] == "bf16" else V_SCALE
    return q.to(torch.bfloat16), quant(k / ks, c["cache"]), quant(v / vs, c["cache"])


def gather_cache(t, c):
    slot, bs = t["slot_mapping"], c["block_size"]
    blk, off = slot // bs, slot % bs
    kc = t["k_cache"][blk, :, :, off, :].reshape(t["T"], c["hkv"], c["head_dim"])
    vc = t["v_cache"][blk, :, :, off]
    return kc, vc


def cos_sim(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def check_accuracy(t, c):
    hq = c["hq"]
    orig = t["packed_qkv"].clone()
    q_ref, k_ref, v_ref = reference(t, c)

    run_op(t, c)
    torch.cuda.synchronize()

    out = t["packed_qkv"].view(t["T"], -1, c["head_dim"])
    kc, vc = gather_cache(t, c)
    _, _, _, step, kv_thr = CACHE_DTYPES[c["cache"]]

    sq, sk, sv = cos_sim(out[:, :hq].float(), q_ref.float()), \
                 cos_sim(kc.float(), k_ref.float()), cos_sim(vc.float(), v_ref.float())
    kv_untouched = torch.equal(out[:, hq:], orig.view_as(out)[:, hq:])
    k_maxerr = (kc.float() - k_ref.float()).abs().max().item()

    ok = sq >= Q_COS_THR and sk >= kv_thr and sv >= kv_thr and kv_untouched
    if step is not None:                       # int8: allow at most 1 LSB on K
        ok &= k_maxerr <= step
    return ok, sq, sk, sv, k_maxerr, kv_untouched


# ---------------- perf ----------------
def effective_bytes(T, c):
    hq, hkv, D, rd = c["hq"], c["hkv"], c["head_dim"], c["rope_dim"]
    cb = CACHE_DTYPES[c["cache"]][2]
    return T * ((hq + hkv) * D * 2          # read Q,K
                + hq * D * 2                # write Q
                + hkv * D * 2               # read V
                + 2 * hkv * D * cb          # write K,V cache
                + 2 * rd * 4                # cos/sin row
                + 8)                        # slot_mapping


def time_fn(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def torch_baseline(t, c):
    q, kq, vq = reference(t, c)
    slot, bs = t["slot_mapping"], c["block_size"]
    blk, off = slot // bs, slot % bs
    t["k_cache"][blk, :, :, off, :] = kq.view(t["T"], c["hkv"], c["head_dim"] // X, X)
    t["v_cache"][blk, :, :, off] = vq
    t["packed_qkv"].view(t["T"], -1, c["head_dim"])[:, :c["hq"]] = q


def bench_case(q_len, batch, cache_len, c, baseline=False):
    t = make_inputs(batch, q_len, cache_len, c)
    T = t["T"]
    ok, sq, sk, sv, kerr, untouched = check_accuracy(t, c)

    ms = time_fn(lambda: run_op(t, c))       # in-place re-run is fine for timing
    gbps = effective_bytes(T, c) / (ms * 1e-3) / 1e9
    path = "tok" if (T >= 4096 and c["rope_dim"] == 128 and c["rope_offset"] % 32 == 0) else "head"

    extra = ""
    if baseline:
        t2 = make_inputs(batch, q_len, cache_len, c)
        ms_ref = time_fn(lambda: torch_baseline(t2, c), warmup=3, rep=10)
        extra = f"  torch {ms_ref * 1e3:9.2f} us  x{ms_ref / ms:6.2f}"

    print(f"[{c['cache']:>6}] q_len={q_len:5d} bs={batch:4d} ctx={cache_len:6d} T={T:6d} "
          f"{path:>4}  cos q/k/v={sq:.6f}/{sk:.6f}/{sv:.6f} kerr={kerr:.3g} "
          f"{'' if untouched else 'KV_MODIFIED '}{'OK' if ok else 'FAIL':4s} "
          f"{ms * 1e3:9.2f} us {gbps:8.1f} GB/s{extra}")
    del t
    torch.cuda.empty_cache()
    return ok, gbps


def run_suite(name, cfg, baseline):
    print("=" * 120)
    print(f"#### suite: {name}")
    suite_pass, summary = True, []
    for c, shapes in expand_cases(cfg):
        tag = (f"{c['src']}->{c['cache']} {c['mode']} Hq/Hkv/D={c['hq']}/{c['hkv']}/{c['head_dim']} "
               f"rope={c['rope_dim']}@{c['rope_offset']} bs={c['block_size']}")
        print(f"---- {tag}  (cos thr Q {Q_COS_THR}, K/V {CACHE_DTYPES[c['cache']][4]}) " + "-" * 10)
        peak, gpass = 0.0, True
        for q_len, batch, cache_len in shapes:
            ok, gbps = bench_case(q_len, batch, cache_len, c, baseline)
            peak, gpass = max(peak, gbps), gpass and ok
        print(f"     peak = {peak:.1f} GB/s   accuracy {'ALL PASS' if gpass else 'FAILURES PRESENT'}")
        summary.append((tag, peak, gpass))
        suite_pass &= gpass
    return suite_pass, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["all", "full", "json"], default="all")
    ap.add_argument("--json-file", default=None,
                    help="op-config json (same schema as JSON_CASES); overrides built-in JSON_CASES")
    ap.add_argument("--baseline", action="store_true", help="also time unfused torch reference")
    args = ap.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}  "
          f"device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"

    json_cfg = JSON_CASES
    if args.json_file:
        with open(args.json_file) as f:
            json_cfg = json.load(f)

    suites = []
    if args.suite in ("all", "full"):
        suites.append(("full", FULL_CASES))
    if args.suite in ("all", "json"):
        suites.append(("json", json_cfg))

    overall_pass, all_summary = True, []
    for name, cfg in suites:
        ok, summary = run_suite(name, cfg, args.baseline)
        overall_pass &= ok
        all_summary += [(name, *s) for s in summary]

    print("=" * 120)
    for name, tag, peak, gpass in all_summary:
        print(f"[{name:>4}] {tag}: peak {peak:8.1f} GB/s  "
              f"({'REACHED' if peak >= TARGET_GBPS else 'below'} single-die target "
              f"{TARGET_GBPS:.0f})  accuracy {'PASS' if gpass else 'FAIL'}")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


def test_fused_rmsnorm_rope_quant_reshape_and_cache():
    ok, _ = run_suite("json", JSON_CASES, baseline=False)
    assert ok


if __name__ == "__main__":
    main()