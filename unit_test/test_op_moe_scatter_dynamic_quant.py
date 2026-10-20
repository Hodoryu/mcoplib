"""Unit test + bandwidth benchmark for mcoplib.op.moe_scatter_dynamic_quant on MetaX C600-U.

Op (12-arg, in-place):
  moe_scatter_dynamic_quant(hidden_status, selected_experts, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale, scatter_tokens_offset,
                            experts_token_count, experts_token_start,
                            experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp)

Semantics (fast path, shared=0):
  For each routed (token -> local expert e) pair one OUTPUT ROW is produced, rows grouped
  by expert. With v = hidden[token].float() * smooth_scale[e] (per channel), bm = max_c|v_c|:
    scatter_per_token_scale[out] = bm * |weight| / QMAX
    scatter_tokens[out]          = round( v * (QMAX/bm) * sign(weight) )
  => dequant  scatter_tokens[out] * scale[out]  ~=  hidden[token] * smooth[e] * weight.

Suites:
  full : dtype=bf16, dst in {int8, fp8}, (128, 8, 8), hidden=4096, all 38 num_tokens
  json : cases from JSON_CASES (or --json-file), same schema as the op-config json
  all  : full + json (default)

Run:
  export CUDA_VISIBLE_DEVICES=10,11
  python unit_test/test_op_moe_scatter_dynamic_quant.py [--suite all|full|json] [--json-file cfg.json]
"""

import argparse
import json
import os
import torch

import mcoplib.op as ops

DEVICE = "cuda"
OP_NAME = "moe_scatter_dynamic_quant"

SHARED_EXPERTS_PER_RANK = 0
SHARED_TOKENS_PER_SP = 0

TARGET_GBPS = 1300.0                                # single-die gate
DUAL_DIE_GBPS = 3480.0                              # dual-die datasheet figure

FLOAT8 = torch.float8_e4m3fn

# src dtype name -> torch dtype
SRC_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}

# dst dtype -> (torch dtype, QMAX, out elem bytes, cos_sim threshold)
DST_DTYPES = {
    "int8": (torch.int8, 127.0, 1, 0.999),
    "fp8":  (FLOAT8,     448.0, 1, 0.999),
}
DST_ALIASES = {"float8_e4m3fn": "fp8", "fp8_e4m3": "fp8"}

# ---------------- suite "full" (original sweep) ----------------
FULL_NUM_TOKENS_CASES = [
    16, 48, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 6144, 8192, 10240,
    12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720,
    32768, 65536,
]
FULL_CASES = {
    OP_NAME: [
        {
            "arg_type": "llm",
            "dtype.dst_dtype": [["bfloat16", "int8"], ["bfloat16", "fp8"]],
            "num_experts.topk.ep_size": [[128, 8, 8]],
            "num_tokens": FULL_NUM_TOKENS_CASES,
            "hidden_size": 4096,
        }
    ]
}

# ---------------- suite "json" (from op-config json) ----------------
JSON_CASES = {
    OP_NAME: [
        {
            "arg_type": "llm",
            "dtype.dst_dtype": [
                ["bfloat16", "int8"]
            ],
            "num_experts.topk.ep_size": [
                [128, 8, 8]
            ],
            "num_tokens": [
                48, 128, 1024, 2048, 4096, 8192, 10240, 16384, 32768
            ],
            "hidden_size": 4096
        }
    ]
}


def expand_cases(cfg):
    """Expand op-config json into groups: (key, cfg_dict, [num_tokens...])."""
    groups = []
    for entry in cfg.get(OP_NAME, []):
        if entry.get("arg_type", "llm") != "llm":
            continue
        hidden = int(entry["hidden_size"])
        for src_name, dst_name in entry["dtype.dst_dtype"]:
            dst_name = DST_ALIASES.get(dst_name, dst_name)
            assert src_name in SRC_DTYPES, f"unsupported dtype {src_name}"
            assert dst_name in DST_DTYPES, f"unsupported dst_dtype {dst_name}"
            for ne, topk, ep in entry["num_experts.topk.ep_size"]:
                epr = ne // ep
                assert topk <= epr, (f"topk={topk} > experts_per_rank={epr}: "
                                     "cannot route to distinct local experts")
                c = dict(src=src_name, dst=dst_name, num_experts=ne, topk=topk,
                         ep_size=ep, experts_per_rank=epr, hidden=hidden)
                groups.append((c, list(entry["num_tokens"])))
    return groups


def make_inputs(num_tokens, c, seed=1234):
    """Each token routes to TOPK *distinct* local experts in [0, experts_per_rank),
    so total routes = num_tokens * topk, all output rows valid."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    epr, topk, hidden = c["experts_per_rank"], c["topk"], c["hidden"]
    src_dtype = SRC_DTYPES[c["src"]]
    dst_dtype = DST_DTYPES[c["dst"]][0]

    h = torch.randn(num_tokens, hidden, dtype=torch.float32, device=DEVICE,
                    generator=g).to(src_dtype)

    rand = torch.rand(num_tokens, epr, device=DEVICE, generator=g)
    selected_experts = rand.argsort(dim=1)[:, :topk].to(torch.int32).contiguous()

    moe_weights = torch.randn(num_tokens, topk, dtype=torch.float32, device=DEVICE, generator=g)

    smooth_scale = (torch.rand(epr, hidden, dtype=torch.float32,
                               device=DEVICE, generator=g) + 0.5).contiguous()

    capacity = num_tokens * topk
    return dict(
        hidden=h, selected_experts=selected_experts, moe_weights=moe_weights,
        smooth_scale=smooth_scale,
        scatter_tokens=torch.zeros(capacity, hidden, dtype=dst_dtype, device=DEVICE),
        scatter_per_token_scale=torch.zeros(capacity, dtype=torch.float32, device=DEVICE),
        scatter_tokens_offset=torch.full((capacity,), -1, dtype=torch.int32, device=DEVICE),
        experts_token_count=torch.zeros(epr + 1, dtype=torch.int32, device=DEVICE),
        experts_token_start=torch.zeros(epr + 1, dtype=torch.int32, device=DEVICE),
        capacity=capacity, experts_per_rank=epr)


def run_op(t):
    ops.moe_scatter_dynamic_quant(
        t["hidden"], t["selected_experts"], t["moe_weights"], t["smooth_scale"],
        t["scatter_tokens"], t["scatter_per_token_scale"], t["scatter_tokens_offset"],
        t["experts_token_count"], t["experts_token_start"],
        t["experts_per_rank"], SHARED_EXPERTS_PER_RANK, SHARED_TOKENS_PER_SP)


def streaming_cosine(t, chunk_rows=2048):
    """Chunked cosine similarity over valid output rows, fp64 accumulation."""
    epr = t["experts_per_rank"]
    count = t["experts_token_count"][:epr].long()
    total_valid = int(count.sum().item())
    start = t["experts_token_start"].long()
    offset = t["scatter_tokens_offset"].long()
    sel = t["selected_experts"].long()
    starts_hi = start[1:epr + 1].contiguous()

    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros((), dtype=torch.float64, device=DEVICE)
    nb = torch.zeros((), dtype=torch.float64, device=DEVICE)

    for lo in range(0, total_valid, chunk_rows):
        hi = min(lo + chunk_rows, total_valid)
        out_ids = torch.arange(lo, hi, device=DEVICE)
        out_expert = torch.searchsorted(starts_hi, out_ids, right=True).clamp_(0, epr - 1)
        out_token = offset[lo:hi]
        match = (sel[out_token] == out_expert[:, None])
        slot = match.float().argmax(dim=1)
        weight = t["moe_weights"][out_token, slot]
        ref = (t["hidden"][out_token].float()
               * t["smooth_scale"][out_expert].float()
               * weight[:, None])
        deq = (t["scatter_tokens"][lo:hi].float()
               * t["scatter_per_token_scale"][lo:hi, None])
        dot += (deq * ref).double().sum()
        na += (deq * deq).double().sum()
        nb += (ref * ref).double().sum()

    denom = (na.sqrt() * nb.sqrt()).clamp_min(1e-30)
    return (dot / denom).item(), total_valid


def effective_bytes(routes, hidden, in_elem_bytes, out_elem_bytes):
    # hidden read + quant write per route, plus 4B per-row scale write
    return routes * (hidden * in_elem_bytes + hidden * out_elem_bytes) + routes * 4


def bench_case(num_tokens, c, warmup=10, rep=50):
    _, _, out_bytes, thr = DST_DTYPES[c["dst"]]
    in_bytes = torch.tensor([], dtype=SRC_DTYPES[c["src"]]).element_size()
    t = make_inputs(num_tokens, c)

    run_op(t)
    torch.cuda.synchronize()
    sim, routes = streaming_cosine(t)

    for _ in range(warmup):
        run_op(t)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_op(t)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    gbps = effective_bytes(routes, c["hidden"], in_bytes, out_bytes) / (median_ms * 1e-3) / 1e9
    ok = "OK" if sim >= thr else "FAIL"
    print(f"[{c['dst']:>4}] T={num_tokens:6d} routes={routes:8d}  "
          f"cos_sim={sim:.6f} {ok:4s}  {median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    del t
    torch.cuda.empty_cache()
    return sim >= thr, gbps


def run_suite(name, cfg):
    print("=" * 100)
    print(f"#### suite: {name}")
    suite_pass, summary = True, []
    for c, num_tokens_list in expand_cases(cfg):
        thr = DST_DTYPES[c["dst"]][3]
        tag = (f"{c['src']}->{c['dst']} E={c['num_experts']} topk={c['topk']} "
               f"ep={c['ep_size']} (epr={c['experts_per_rank']}) hidden={c['hidden']}")
        print(f"---- {tag}  (cos_sim threshold {thr}) " + "-" * 20)
        peak, gpass = 0.0, True
        for T in num_tokens_list:
            ok, gbps = bench_case(T, c)
            peak = max(peak, gbps)
            gpass &= ok
        print(f"     peak = {peak:.1f} GB/s   accuracy {'ALL PASS' if gpass else 'FAILURES PRESENT'}")
        summary.append((tag, peak, gpass))
        suite_pass &= gpass
    return suite_pass, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["all", "full", "json"], default="all")
    ap.add_argument("--json-file", default=None,
                    help="op-config json (same schema as JSON_CASES); overrides built-in JSON_CASES")
    args = ap.parse_args()

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
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
        ok, summary = run_suite(name, cfg)
        overall_pass &= ok
        all_summary += [(name, *s) for s in summary]

    print("=" * 100)
    for name, tag, peak, gpass in all_summary:
        print(f"[{name:>4}] {tag}: peak {peak:8.1f} GB/s  "
              f"({'REACHED' if peak >= TARGET_GBPS else 'below'} single-die target "
              f"{TARGET_GBPS:.0f})  accuracy {'PASS' if gpass else 'FAIL'}")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


if __name__ == "__main__":
    main()