# SPDX-License-Identifier: Apache-2.0
"""
gumbel_sample 精度 + 性能测试
"""
import argparse
import importlib

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

NUM_TOKENS = 32
VOCAB_SIZE = 151936
MAX_NUM_REQS = 16
DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=25)
parser.add_argument("--rep", type=int, default=200)
args = parser.parse_args()

from mcoplib.triton_gumbel_sample import (
    gumbel_sample, tl_rand32, tl_rand64, _DRAFT_NOISE_SALT
)

SALT = _DRAFT_NOISE_SALT
from vllm.triton_utils import tldevice  # noqa: E402


# ============================================================================
# 固定输入（手写 + 固定种子）
# ============================================================================
def make_inputs(dtype=torch.float32, with_invalid_row=True):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    logits = (torch.randn(NUM_TOKENS, VOCAB_SIZE, generator=g, device=DEVICE) * 3.0).to(dtype)

    # 每个请求槽位一个温度，含 0（贪心）、1（不缩放）和其他值
    temperature = torch.tensor(
        [0.0, 0.5, 1.0, 2.0, 0.7, 1.3, 0.0, 0.9,
         1.0, 0.3, 1.5, 0.8, 0.0, 1.1, 0.6, 2.5],
        dtype=torch.float32, device=DEVICE)
    seed = torch.tensor(
        [11, 22, 33, 44, 55, 66, 77, 88,
         99, 1234567, 2**31 - 1, 2**40 + 7, 5, 6, 7, 8],
        dtype=torch.int64, device=DEVICE)

    # 32 行 -> 16 个槽位：前 8 个槽位各占 3 行（模拟投机解码），其余各占 1 行
    mapping = []
    for r in range(8):
        mapping += [r] * 3
    mapping += list(range(8, 16))
    assert len(mapping) == NUM_TOKENS
    if with_invalid_row:
        mapping[-1] = -1  # 无效行
    expanded_idx_mapping = torch.tensor(mapping, dtype=torch.int32, device=DEVICE)

    # pos：同一请求的连续行 pos 连续
    pos_list, col_list, cnt = [], [], {}
    for r in mapping:
        k = cnt.get(r, 0)
        cnt[r] = k + 1
        pos_list.append(100 + 17 * max(r, 0) + k)
        col_list.append(k)
    pos = torch.tensor(pos_list, dtype=torch.int64, device=DEVICE)
    cache_col = torch.tensor(col_list, dtype=torch.int32, device=DEVICE)
    return logits, expanded_idx_mapping, temperature, seed, pos, cache_col


# ============================================================================
# 参考实现：用相同的 Triton 噪声函数导出 "logits/T + noise"，在 torch 里做 argmax
# ============================================================================
@triton.jit
def _dump_scores_kernel(out_ptr, out_stride, logits_ptr, logits_stride,
                        idx_map_ptr, seeds_ptr, pos_ptr, temp_ptr, vocab_size,
                        BLOCK: tl.constexpr, IS_DRAFTING: tl.constexpr,
                        USE_FP64: tl.constexpr, APPLY_TEMPERATURE: tl.constexpr):
    token_idx = tl.program_id(0).to(tl.int64)
    bid = tl.program_id(1)
    block = bid * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab_size
    x = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask,
                other=float("-inf")).to(tl.float32)
    req = tl.load(idx_map_ptr + token_idx).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + token_idx)
    if temp != 0.0 and APPLY_TEMPERATURE:
        x = x / temp
    if USE_FP64:
        x = x.to(tl.float64)
    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + SALT
        gs = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gs, block, includes_zero=False)
            n = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gs, block, includes_zero=False)
            n = -tl.log(-tldevice.log1p(-u))
        x = tl.where(mask, x + n, float("-inf"))
    tl.store(out_ptr + token_idx * out_stride + block, x, mask=mask)


def reference(logits, idx_map, temp, seed, pos, apply_temperature,
              is_drafting, use_fp64):
    out = torch.empty(logits.shape, device=DEVICE,
                      dtype=torch.float64 if use_fp64 else torch.float32)
    BLOCK = 1024  # 必须与原实现一致（keys = 全局 token id，与 BLOCK 无关，但保持一致）
    grid = (logits.shape[0], triton.cdiv(logits.shape[1], BLOCK))
    _dump_scores_kernel[grid](out, out.stride(0), logits, logits.stride(0),
                              idx_map, seed, pos, temp, logits.shape[1],
                              BLOCK=BLOCK, IS_DRAFTING=is_drafting,
                              USE_FP64=use_fp64,
                              APPLY_TEMPERATURE=apply_temperature)
    return out.argmax(dim=-1), out  # torch argmax：并列时取第一个，和 tl.max 一致


# ============================================================================
# 精度测试
# ============================================================================
def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    return ok


def accuracy_tests():
    print("=" * 78)
    print("精度测试")
    print("=" * 78)
    all_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        for use_fp64 in (False, True):
            for is_drafting in (False, True):
                logits, m, t, s, p, _ = make_inputs(dtype)
                out = gumbel_sample(logits, m, t, s, p, apply_temperature=True,
                                    is_drafting=is_drafting, use_fp64=use_fp64)
                ref, _ = reference(logits, m, t, s, p, True, is_drafting, use_fp64)
                n_mis = (out != ref).sum().item()
                all_ok &= check(
                    f"bitwise vs ref  dtype={str(dtype)[6:]:8s} fp64={use_fp64!s:5s} "
                    f"draft={is_drafting!s:5s}", n_mis == 0,
                    f"mismatch={n_mis}/{NUM_TOKENS}")

    logits, m, t, s, p, col = make_inputs()
    out1 = gumbel_sample(logits, m, t, s, p, True, False)
    out2 = gumbel_sample(logits, m, t, s, p, True, False)
    all_ok &= check("确定性（两次调用结果相同）", torch.equal(out1, out2))

    # 2) 贪心行(T=0 或无效行) == torch.argmax
    req = m.long()
    greedy = (req < 0) | (t[req.clamp(min=0)] == 0)
    greedy_ok = torch.equal(out1[greedy], logits.argmax(-1)[greedy])
    all_ok &= check("T=0 / 无效行 == torch.argmax", greedy_ok,
                    f"(rows={greedy.sum().item()})")

    # 3) drafting 噪声与 target 不同
    out_d = gumbel_sample(logits, m, t, s, p, True, True)
    diff = (out_d != out1)[~greedy].float().mean().item()
    all_ok &= check("draft/target 噪声独立", diff > 0.5, f"diff_rate={diff:.2f}")

    # 4) logits_cache 存的是未除温度的原始 logits，且无效行不写
    cache = torch.full((MAX_NUM_REQS, 3, VOCAB_SIZE), 7.0, device=DEVICE)
    gumbel_sample(logits, m, t, s, p, True, False, logits_cache=cache,
                  logits_cache_col=col)
    ok = True
    for r in range(NUM_TOKENS):
        rq = m[r].item()
        if rq >= 0:
            ok &= torch.equal(cache[rq, col[r].item()], logits[r])
    ok &= bool((cache[15] == 7.0).all())  # 槽位 15 只对应无效行
    all_ok &= check("logits_cache 内容正确", ok)

    # 5) fp32 vs fp64 一致率（不要求 100%）
    o32 = gumbel_sample(logits, m, t, s, p, True, False, use_fp64=False)
    o64 = gumbel_sample(logits, m, t, s, p, True, False, use_fp64=True)
    print(f"  [INFO] fp32 vs fp64 一致率 = {(o32 == o64).float().mean().item():.3f}")

    # 6) 分布检验：经验频率 vs softmax(logits/T)
    T, N, K = 0.8, 65536, 16
    lg = torch.full((1, VOCAB_SIZE), float("-inf"), device=DEVICE)
    hot = torch.arange(0, VOCAB_SIZE, VOCAB_SIZE // K, device=DEVICE)[:K]
    lg[0, hot] = torch.linspace(-2, 2, K, device=DEVICE)
    lgN = lg.expand(N, -1).contiguous()
    mm = torch.zeros(N, dtype=torch.int32, device=DEVICE)
    tt = torch.tensor([T], device=DEVICE)
    ss = torch.tensor([2024], dtype=torch.int64, device=DEVICE)
    pp = torch.arange(N, dtype=torch.int64, device=DEVICE)
    smp = gumbel_sample(lgN, mm, tt, ss, pp, True, False)
    emp = torch.bincount(smp, minlength=VOCAB_SIZE)[hot].double() / N
    prob = torch.softmax(lg[0, hot].double() / T, -1)
    tv = 0.5 * (emp - prob).abs().sum().item()
    all_ok &= check("分布 ≈ softmax(logits/T)", tv < 0.01, f"TV={tv:.4f}")
    del lgN

    print(f"\n精度结果: {'ALL PASS' if all_ok else 'HAS FAILURES'}\n")
    return all_ok


# ============================================================================
# 性能测试
# ============================================================================
def bytes_moved(dtype, use_cache, use_fp64):
    es = torch.tensor([], dtype=dtype).element_size()
    nb = triton.cdiv(VOCAB_SIZE, 1024)
    b = NUM_TOKENS * VOCAB_SIZE * es                     # 读 logits
    if use_cache:
        b += NUM_TOKENS * VOCAB_SIZE * es                # 写 cache
    b += NUM_TOKENS * nb * (8 + (8 if use_fp64 else 4)) * 2  # 写+读 局部结果
    b += NUM_TOKENS * (4 + 8 + 4 + 8 + 8)                # mapping/pos/temp/seed/out
    return b


def perf_tests():
    print("=" * 78)
    print(f"性能测试  num_tokens={NUM_TOKENS} vocab={VOCAB_SIZE}  "
          f"GPU={torch.cuda.get_device_name()}")
    print("=" * 78)
    print(f"{'dtype':6s} {'fp64':5s} {'cache':5s} {'draft':5s} | "
          f"{'median(us)':>10s} {'p20(us)':>9s} {'p80(us)':>9s} | {'GB/s':>8s}")
    print("-" * 78)
    for dtype in (torch.float32, torch.bfloat16):
        for use_cache in (False, True):
            for use_fp64 in (False, True):
                logits, m, t, s, p, col = make_inputs(dtype)
                cache = (torch.empty(MAX_NUM_REQS, 3, VOCAB_SIZE, dtype=dtype,
                                     device=DEVICE) if use_cache else None)
                fn = lambda: gumbel_sample(  # noqa: E731
                    logits, m, t, s, p, True, False,
                    logits_cache=cache,
                    logits_cache_col=col if use_cache else None,
                    use_fp64=use_fp64)
                # do_bench 每次计时前会清空 L2，避免 19MB logits 常驻 L2 导致带宽虚高
                ms = do_bench(fn, warmup=args.warmup, rep=args.rep,
                              quantiles=[0.5, 0.2, 0.8])
                med, p20, p80 = ms
                gbs = bytes_moved(dtype, use_cache, use_fp64) / (med * 1e-3) / 1e9
                print(f"{str(dtype)[6:]:6s} {use_fp64!s:5s} {use_cache!s:5s} "
                      f"{'False':5s} | {med*1e3:10.2f} {p20*1e3:9.2f} "
                      f"{p80*1e3:9.2f} | {gbs:8.1f}")
    print("-" * 78)
    print("all_greddy")
    for dtype in (torch.float32, torch.bfloat16):
        for use_cache in (False, True):
            for use_fp64 in (False, True):
                logits, m, t, s, p, col = make_inputs(dtype)
                cache = (torch.empty(MAX_NUM_REQS, 3, VOCAB_SIZE, dtype=dtype,
                                     device=DEVICE) if use_cache else None)
                fn = lambda: gumbel_sample(  # noqa: E731
                    logits, m, t, s, p, True, False,
                    logits_cache=cache,
                    logits_cache_col=col if use_cache else None,
                    use_fp64=use_fp64, all_greedy=True)
                # do_bench 每次计时前会清空 L2，避免 19MB logits 常驻 L2 导致带宽虚高
                ms = do_bench(fn, warmup=args.warmup, rep=args.rep,
                              quantiles=[0.5, 0.2, 0.8])
                med, p20, p80 = ms
                gbs = bytes_moved(dtype, use_cache, use_fp64) / (med * 1e-3) / 1e9
                print(f"{str(dtype)[6:]:6s} {use_fp64!s:5s} {use_cache!s:5s} "
                      f"{'False':5s} | {med*1e3:10.2f} {p20*1e3:9.2f} "
                      f"{p80*1e3:9.2f} | {gbs:8.1f}")

    # 基线：纯读 logits 的 torch.argmax，作为"访存上限"的参照
    print("-" * 78)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    ok = accuracy_tests()
    perf_tests()
    raise SystemExit(0 if ok else 1)
