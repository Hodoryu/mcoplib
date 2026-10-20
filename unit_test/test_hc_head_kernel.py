import argparse
import torch
import triton
import triton.language as tl
import mcoplib
import mcoplib._C
import mcoplib.sgl_kernel


# ============================ Triton 标杆 ============================
@triton.jit
def _hc_head_kernel(
    x_ptr, fn_ptr, scale_ptr, base_ptr, y_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    sumsq = tl.zeros((), dtype=tl.float32)
    mix = tl.zeros((HC_MULT,), dtype=tl.float32)
    x_row = x_ptr + pid * K_TOTAL
    m_idx = tl.arange(0, HC_MULT)

    for k_off in tl.range(0, K_TOTAL, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL
        x_tile = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_tile * x_tile, axis=0)
        fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
        fn_mask = (m_idx[:, None] < HC_MULT) & k_mask[None, :]
        fn_tile = tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0)
        mix += tl.sum(fn_tile * x_tile[None, :], axis=1)

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_v = tl.load(scale_ptr).to(tl.float32)
    base_v = tl.load(base_ptr + m_idx).to(tl.float32)
    pre = tl.sigmoid(mix * rsqrt * scale_v + base_v) + hc_eps

    y_row = y_ptr + pid * hidden_size
    for d_off in tl.range(0, hidden_size, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_offs < hidden_size
        x_offs = m_idx[:, None] * hidden_size + d_offs[None, :]
        x_mask = (m_idx[:, None] < HC_MULT) & d_mask[None, :]
        x_block = tl.load(x_row + x_offs, mask=x_mask, other=0.0).to(tl.float32)
        y_block = tl.sum(pre[:, None] * x_block, axis=0)
        tl.store(y_row + d_offs, y_block.to(y_ptr.dtype.element_ty), mask=d_mask)


def triton_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps: float, hc_eps: float):
    """Triton 标杆实现。注意：hc_mult 非 2 的幂时会越界，调用方需跳过。"""
    assert x.is_contiguous() and hc_fn.is_contiguous() and x.dim() == 3
    T, hc_mult, hidden_size = x.shape
    y = torch.empty((T, hidden_size), dtype=x.dtype, device=x.device)
    if T == 0:
        return y
    _hc_head_kernel[(T,)](
        x, hc_fn, hc_scale, hc_base, y,
        hidden_size=hidden_size,
        HC_MULT=max(1, triton.next_power_of_2(hc_mult)),
        K_TOTAL=hc_mult * hidden_size,
        BLOCK_K=512, BLOCK_D=512,
        norm_eps=norm_eps, hc_eps=hc_eps,
        num_warps=4,
    )
    return y


# ============================ CUDA 被测实现 ============================
def fused_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps: float, hc_eps: float,
                  out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
    torch.ops.sgl_kernel.fused_hc_head.default(
        out, x, hc_fn, hc_scale, hc_base, float(norm_eps), float(hc_eps))
    return out


# ============================ fp32 真值 ============================
def torch_golden_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    """全 fp32 计算，返回 fp32（不做 dtype 截断），作为精度真值。"""
    T, M, H = x.shape
    xf = x.float()
    flat = xf.reshape(T, M * H)
    rs = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    mix = flat @ hc_fn.t()
    pre = torch.sigmoid(mix * rs * hc_scale + hc_base) + hc_eps
    return (pre.unsqueeze(-1) * xf).sum(dim=1)


# ============================ 工具 ============================
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TOL = {torch.bfloat16: (2e-2, 2e-2), torch.float16: (5e-3, 5e-3), torch.float32: (1e-4, 1e-4)}


def make_inputs(T, M, H, dtype, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, M, H, dtype=torch.float32, device=device, generator=g).to(dtype)
    hc_fn = torch.randn(M, M * H, dtype=torch.float32, device=device, generator=g) * (1.0 / (M * H) ** 0.5)
    hc_scale = torch.rand(1, dtype=torch.float32, device=device, generator=g) + 0.5
    hc_base = torch.randn(M, dtype=torch.float32, device=device, generator=g) * 0.1
    return x, hc_fn, hc_scale, hc_base


def compare(a, b, dtype):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    rel = (diff.norm() / b.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    atol, rtol = TOL[dtype]
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    return ok, max_abs, rel, cos


def triton_supported(M):
    return M == triton.next_power_of_2(M)


# ============================ 精度 ============================
def check_accuracy(T, M, H, dtype, norm_eps, hc_eps):
    x, fn, s, b = make_inputs(T, M, H, dtype)
    y_cu = fused_hc_head(x, fn, s, b, norm_eps, hc_eps)
    y_gd = torch_golden_hc_head(x, fn, s, b, norm_eps, hc_eps)
    y_tr = triton_hc_head(x, fn, s, b, norm_eps, hc_eps) if triton_supported(M) else None
    torch.cuda.synchronize()

    res = {"cu_vs_gold": compare(y_cu, y_gd, dtype)}
    if y_tr is not None:
        res["tri_vs_gold"] = compare(y_tr, y_gd, dtype)
        res["cu_vs_tri"] = compare(y_cu, y_tr, dtype)
    return res


# ============================ 性能 ============================
def bench(T, M, H, dtype, norm_eps, hc_eps, peak_gbs):
    x, fn, s, b = make_inputs(T, M, H, dtype)
    out = torch.empty((T, H), dtype=dtype, device=x.device)
    esz = x.element_size()

    t_cu = triton.testing.do_bench(
        lambda: fused_hc_head(x, fn, s, b, norm_eps, hc_eps, out=out), warmup=25, rep=100)  # ms
    t_tr = (triton.testing.do_bench(
        lambda: triton_hc_head(x, fn, s, b, norm_eps, hc_eps), warmup=25, rep=100)
        if triton_supported(M) else float("nan"))

    # 最小必要访存：x 读 1 次 + y 写 1 次 + hc_fn/base/scale 各 1 次
    bytes_min = T * M * H * esz + T * H * esz + M * M * H * 4 + M * 4 + 4
    bw_cu = bytes_min / (t_cu * 1e-3) / 1e9
    bw_tr = bytes_min / (t_tr * 1e-3) / 1e9
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA
    #     ],
    #     with_stack=True,  # 关键：记录调用栈
    #     record_shapes=True
    # ) as prof:
    #     fused_hc_head(x, fn, s, b, norm_eps, hc_eps, out=out)

    # # 打印完整的内核名称
    # print("--- 完整内核名称 ---")
    # for event in prof.events():
    #     # 检查事件是否包含内核信息
    #     if hasattr(event, 'kernels') and event.kernels:
    #         for kernel in event.kernels:
    #             # kernel.name 就是底层 CUDA 内核的名称
    #             print(f"Kernel Name: {kernel.name}")

    return dict(
        t_cu=t_cu * 1e3, t_tr=t_tr * 1e3,               # us
        bw_cu=bw_cu, bw_tr=bw_tr,
        util_cu=bw_cu / peak_gbs * 100, util_tr=bw_tr / peak_gbs * 100,
        speedup=t_tr / t_cu,
        t_ideal=bytes_min / (peak_gbs * 1e9) * 1e6,     # us
    )


# ============================ main ============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument("--tokens", type=int, nargs="+",
                   default=[1024 * i for i in range(1, 9)])   # 1k..8k
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--norm-eps", type=float, default=1e-6)
    p.add_argument("--hc-eps", type=float, default=1e-6)
    p.add_argument("--peak-bw", type=float, default=1500.0, help="峰值带宽 GB/s")
    p.add_argument("--skip-acc", action="store_true")
    p.add_argument("--skip-perf", action="store_true")
    a = p.parse_args()

    dtype = DTYPES[a.dtype]
    M, H = a.hc_mult, a.hidden
    print(f"GPU={torch.cuda.get_device_name()}  hidden={H}  hc_mult={M}  dtype={a.dtype}  "
          f"peak={a.peak_bw:.0f}GB/s  (TP=8: per-rank shape identical)")
    if not triton_supported(M):
        print(f"[warn] hc_mult={M} 非 2 的幂，Triton 标杆会越界，已跳过 Triton 对比")

    if not a.skip_acc:
        print("\n== Accuracy (golden = torch fp32) ==")
        hdr = f"{'T':>6} | {'case':>11} | {'ok':>5} | {'max_abs':>10} | {'rel_l2':>10} | {'cos':>10}"
        print(hdr)
        print("-" * len(hdr))
        all_ok = True
        for T in [1, 3, 17, 255, 513] + a.tokens:   # 含边界：小 T、非 TOK 整倍数
            res = check_accuracy(T, M, H, dtype, a.norm_eps, a.hc_eps)
            for name, (ok, ma, rel, cos) in res.items():
                if name == "cu_vs_gold":
                    all_ok &= ok
                print(f"{T:>6} | {name:>11} | {str(ok):>5} | {ma:>10.3e} | {rel:>10.3e} | {cos:>10.7f}")
        print(f"\nCUDA accuracy: {'PASS' if all_ok else 'FAIL'}")

    if not a.skip_perf:
        print("\n== Performance (BW = bytes_min / time) ==")
        hdr = (f"{'T':>6} | {'cuda(us)':>9} | {'triton(us)':>10} | {'ideal(us)':>9} | {'speedup':>8} | "
               f"{'cuda GB/s':>9} | {'cuda %pk':>8} | {'tri GB/s':>9} | {'tri %pk':>8}")
        print(hdr)
        print("-" * len(hdr))
        for T in a.tokens:
            r = bench(T, M, H, dtype, a.norm_eps, a.hc_eps, a.peak_bw)
            print(f"{T:>6} | {r['t_cu']:>9.2f} | {r['t_tr']:>10.2f} | {r['t_ideal']:>9.2f} | "
                  f"{r['speedup']:>7.2f}x | {r['bw_cu']:>9.1f} | {r['util_cu']:>7.1f}% | "
                  f"{r['bw_tr']:>9.1f} | {r['util_tr']:>7.1f}%")


if __name__ == "__main__":
    main()
