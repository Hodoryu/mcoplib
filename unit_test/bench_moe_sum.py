# SPDX-License-Identifier: Apache-2.0
"""Step5 `moe_sum` 基线采集 / 性能验收脚本。

只测 ``torch.ops._moe_C.moe_sum``（实际执行 ``moe_sum_vec_kernel``），不做算子间对比。

* ``--correct``  与 fp32 参考做正确性回归（含 pad-aware、非对齐 H、T=0）；
* ``--perf``     采集 us / GB/s；
* ``--ep-size``  >0 时额外测 EP 场景（2 参非 pad-aware vs 4 参 pad-aware），
  如 ``--ep-size 32``；
* ``--json``     结果落盘；
* ``--baseline`` 与已有 json 对比，打印加速比（优化后复测用）。

默认生产 shape：H=4096, topk=8, bf16。

运行（需 CUDA/MACA）：

    python bench_moe_sum.py --correct
    python bench_moe_sum.py --perf --json base.json
    python bench_moe_sum.py --perf --baseline base.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from mcoplib.triton_utils import tl, triton

@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )

def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


HIDDEN = 4096
TOPK = 8
DTYPE = torch.bfloat16

TOKEN_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
NUM_EXPERTS = 352  # config_step5_text.json: moe_num_experts

def _import_extensions() -> list[str]:
    loaded = []
    for name in (
        "mcoplib._moe_C",
        "vllm_metax._moe_C_stable_libtorch",
        "vllm._moe_C",
        "vllm",
    ):
        try:
            __import__(name)
            loaded.append(name)
        except Exception:
            pass
    return loaded


def get_moe_sum():
    fn = getattr(getattr(torch.ops, "_moe_C", None), "moe_sum", None)
    if fn is None:
        raise RuntimeError("torch.ops._moe_C.moe_sum 未注册：先 import 对应扩展")
    return fn


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------
def perf_cases():
    for T in TOKEN_SIZES:
        yield f"T={T}", T


def correct_cases(hidden: int, topk: int):
    for T in (1, 3, 8, 64, 511, 1024, 4096, 16384):
        yield f"T={T}", T, hidden
    # 非对齐 H -> 走 scalar 回退
    if hidden % 8 != 0:
        yield "non-aligned H", 16, hidden
    yield "H not % 8", 16, hidden - (hidden % 8) + 1


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
def _iters_for(T: int, hidden: int, topk: int, target_s: float = 0.03) -> int:
    bytes_per_call = (T * topk * hidden + T * hidden) * 2
    est_s = bytes_per_call / 500e9
    return max(20, min(2000, int(target_s / max(est_s, 1e-7))))


def time_ms(fn, iters: int, warmup: int = 25, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    samples.sort()
    return samples[len(samples) // 2]


def _gbps(bytes_moved: float, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / 1e9


# --------------------------------------------------------------------------
# 正确性
# --------------------------------------------------------------------------
def _ref_moe_sum(x: torch.Tensor, ids=None, emap=None) -> torch.Tensor:
    xf = x.float()
    if ids is None:
        return xf.sum(dim=1).to(x.dtype)
    keep = ids >= 0
    if emap is not None:
        mapped = emap[ids.clamp(min=0).to(torch.int64)]
        keep = keep & (mapped >= 0)
    return (xf * keep.unsqueeze(-1)).sum(dim=1).to(x.dtype)


def make_ep_inputs(
    T: int, topk: int, hidden: int, num_experts: int, ep_size: int, device: str,
    dtype, seed: int = 0,
):
    """构造 EP 场景：本 rank 只写自己的专家行，其余 slot 在 cache 里是 0。

    expert_map 按 ep_rank=0 连续切分（同 expert_map_manager.py:69），
    返回 (x, topk_ids, expert_map, 本 rank 专家数, 有效 slot 占比)。
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    base = num_experts // ep_size  # EP32: 352 // 32 = 11
    emap = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    emap[:base] = torch.arange(base, dtype=torch.int32, device=device)

    ids = torch.randint(
        0, num_experts, (T, topk), generator=gen, device=device, dtype=torch.int32
    )
    x = torch.randn(
        T, topk, hidden, generator=gen, device=device, dtype=dtype
    )
    local = emap[ids.to(torch.int64)] >= 0  # [T, topk]
    x = x * local.unsqueeze(-1).to(dtype)  # 非本 rank 的行置 0
    valid = float(local.float().mean().item()) if T else 0.0
    return x, ids, emap, base, valid


def run_correct(hidden: int, topk: int, device: str, dtype) -> int:
    moe_sum = get_moe_sum()
    failures = 0
    print(f"correctness: hidden={hidden} topk={topk} dtype={dtype}")

    for name, T, H in correct_cases(hidden, topk):
        gen = torch.Generator(device=device).manual_seed(7)
        x = torch.randn(T, topk, H, device=device, dtype=dtype)
        out = torch.empty(T, H, device=device, dtype=dtype)
        if T == 0:
            moe_sum(x, out)
            torch.cuda.synchronize()
            ok = out.shape == (0, H)
            print(f"[{'ok  ' if ok else 'FAIL'}] {name:14s} (T=0)")
            failures += 0 if ok else 1
            continue

        moe_sum(x, out)
        ref = _ref_moe_sum(x)
        ok = torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=6e-2)
        print(f"[{'ok  ' if ok else 'FAIL'}] {name:14s} T={T} H={H}")
        if not ok:
            failures += 1
            print(f"        max diff={float((out.float() - ref.float()).abs().max()):.3e}")

        # pad-aware 分支
        ids = torch.arange(topk, dtype=torch.int32, device=device).expand(T, topk).contiguous()
        ids = ids.clone()
        if T > 1:
            ids[0, 0] = -1
        emap = torch.arange(max(topk, 8), dtype=torch.int32, device=device)
        if topk > 1:
            emap[-1] = -1
        out.zero_()
        moe_sum(x, out, ids, emap)
        ref_pad = _ref_moe_sum(x, ids, emap)
        ok = torch.allclose(out.float(), ref_pad.float(), rtol=1e-2, atol=6e-2)
        print(f"[{'ok  ' if ok else 'FAIL'}] {name:14s} pad-aware")
        if not ok:
            failures += 1
            print(f"        max diff={float((out.float() - ref_pad.float()).abs().max()):.3e}")

    print(f"\ncorrectness: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return failures


def run_correct_ep(hidden, topk, device, dtype, num_experts, ep_size) -> int:
    """EP 场景：2 参（非 pad-aware）与 4 参（pad-aware）必须给出相同结果。"""
    moe_sum = get_moe_sum()
    failures = 0
    print(f"\nEP{ep_size} correctness: num_experts={num_experts} "
          f"local/rank={num_experts // ep_size}")
    for T in (1, 8, 64, 512, 4096):
        x, ids, emap, base, valid = make_ep_inputs(
            T, topk, hidden, num_experts, ep_size, device, dtype
        )
        ref = _ref_moe_sum(x, ids, emap)
        out2 = torch.empty(T, hidden, device=device, dtype=dtype)
        out4 = torch.empty(T, hidden, device=device, dtype=dtype)
        moe_sum(x, out2)                 # 非 pad-aware：把 0 slot 也加进去
        moe_sum(x, out4, ids, emap)      # pad-aware：跳过 0 slot
        ok2 = torch.allclose(out2.float(), ref.float(), rtol=1e-2, atol=6e-2)
        ok4 = torch.allclose(out4.float(), ref.float(), rtol=1e-2, atol=6e-2)
        ok = ok2 and ok4
        print(f"[{'ok  ' if ok else 'FAIL'}] T={T:5d} 有效 slot={valid * 100:5.2f}% "
              f"2参={ok2} 4参={ok4}")
        if not ok:
            failures += 1
    print(f"EP correctness: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return failures


# --------------------------------------------------------------------------
# 性能
# --------------------------------------------------------------------------
def run_perf(hidden: int, topk: int, device: str, dtype, json_path, baseline) -> list[dict]:
    moe_sum = get_moe_sum()
    # moe_sum = moe_sum_reduce_triton
    print(
        f"perf: hidden={hidden} topk={topk} dtype={dtype} | T={TOKEN_SIZES}"
    )

    base_rows = {}
    if baseline:
        with open(baseline, encoding="utf-8") as fh:
            base = json.load(fh)
        base_rows = {(r["case"], r["T"]): r for r in base.get("rows", [])}
        print(f"[baseline] {baseline}")

    rows = []
    header = "| T | 场景 | us | GB/s |"
    sep = "|---:|---|---:|---:|"
    if base_rows:
        header += " baseline us | 加速比 |"
        sep += "|---:|---:|"
    print(header)
    print(sep)

    for name, T in perf_cases():
        gen = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(
            T, topk, hidden, generator=gen, device=device, dtype=dtype
        )
        out = torch.empty(T, hidden, device=device, dtype=dtype)
        iters = _iters_for(T, hidden, topk)
        ms = time_ms(lambda: moe_sum(x, out), iters)
        bytes_moved = (T * topk * hidden + T * hidden) * 2
        row = {"case": name, "T": T, "us": ms * 1e3, "gbps": _gbps(bytes_moved, ms)}
        rows.append(row)

        line = f"| {T} | {name} | {row['us']:.2f} | {row['gbps']:.0f} |"
        b = base_rows.get((name, T))
        if base_rows:
            if b:
                line += f" {b['us']:.2f} | {b['us'] / row['us']:.2f}x |"
            else:
                line += " -- | -- |"
        print(line)

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config": {"hidden": hidden, "topk": topk, "dtype": str(dtype)},
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"[saved] {json_path}")
    return rows


def run_perf_ep(hidden, topk, device, dtype, num_experts, ep_size) -> list[dict]:
    """EP 场景：非 pad-aware（2 参）vs pad-aware（4 参），同一份稀疏输入。"""
    moe_sum = get_moe_sum()
    base_local = num_experts // ep_size
    print(f"\n[EP{ep_size}] num_experts={num_experts} -> local/rank={base_local}；"
          f"期望有效 slot ≈ {topk * base_local / num_experts:.2f}/{topk}")
    print("| T | 2参(non-pad) us | 2参 GB/s | 4参(pad-aware) us | 4参 GB/s |"
          " 加速比 | 有效 slot |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for _, T in perf_cases():
        x, ids, emap, _, valid = make_ep_inputs(
            T, topk, hidden, num_experts, ep_size, device, dtype
        )
        out = torch.empty(T, hidden, device=device, dtype=dtype)
        iters = _iters_for(T, hidden, topk)

        ms2 = time_ms(lambda: moe_sum(x, out, emap), iters)
        ms4 = time_ms(lambda: moe_sum(x, out, ids, emap), iters)
        bytes_moved = (T * topk * hidden + T * hidden) * 2

        rows.append(
            {"T": T, "us_2arg": ms2 * 1e3, "us_4arg": ms4 * 1e3, "valid_frac": valid}
        )
        print(
            f"| {T} | {ms2 * 1e3:.2f} | {_gbps(bytes_moved, ms2):.0f} |"
            f" {ms4 * 1e3:.2f} | {_gbps(bytes_moved, ms4):.0f} |"
            f" {ms2 / ms4:.2f}x | {valid * 100:.2f}% |"
        )
    return rows


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correct", action="store_true", help="只跑正确性")
    ap.add_argument("--perf", action="store_true", help="只跑性能")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--json", default=None, help="结果落盘路径")
    ap.add_argument("--baseline", default=None, help="基线 json，打印加速比")
    ap.add_argument(
        "--ep-size", type=int, default=0,
        help=">0 时额外测 EP 场景（如 32 = EP32）",
    )
    ap.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[error] 需要 CUDA/MACA 设备", file=sys.stderr)
        return 2

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = "cuda"

    loaded = _import_extensions()
    print(f"[ext] loaded: {loaded or '(none)'}")

    do_correct = args.correct or not args.perf
    do_perf = args.perf or not args.correct

    failures = 0
    if do_correct:
        failures = run_correct(args.hidden, args.topk, device, dtype)
        if args.ep_size > 0:
            failures += run_correct_ep(
                args.hidden, args.topk, device, dtype, args.num_experts, args.ep_size
            )
    if do_perf:
        print()
        run_perf(args.hidden, args.topk, device, dtype, args.json, args.baseline)
        if args.ep_size > 0:
            run_perf_ep(
                args.hidden, args.topk, device, dtype, args.num_experts, args.ep_size
            )

    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
