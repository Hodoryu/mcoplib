"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

Unit test for mcoplib.op.rotary_embedding (default op).

Test cases are driven by ROTARY_EMBEDDING_CONFIGS (same format as the
benchmark JSON config "rotary_embedding").
"""

import os
import sys
import random
import itertools
import torch
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from mcoplib.op import rotary_embedding
from measure_cuda import measure_cuda


# ---------------------------------------------------------------------------
# Test configs (copied from the JSON spec)
# ---------------------------------------------------------------------------
ROTARY_EMBEDDING_CONFIGS = [
    {
        "arg_type": "llm",
        "attn_mode": "prefill",
        "dtype": "bfloat16",
        "q_head_num.kv_head_num.head_dim": [
            [80, 8, 128]
        ],
        "rope_offset.rope_dim": [
            [0, 128]
        ],
        "batch_size.cache_len.q_len": [
            [1, 0, 1024],
            [1, 0, 2048],
            [1, 0, 4096],
            [1, 0, 8192],
            [1, 0, 10240],
            [1, 0, 16384],
            [1, 0, 32768],
            [1, 4096, 4096],
            [1, 5120, 5120],
            [1, 8192, 8192],
            [1, 16384, 4096],
            [1, 28672, 4096]
        ]
    },
    {
        "arg_type": "llm",
        "attn_mode": "decode",
        "dtype": "bfloat16",
        "q_head_num.kv_head_num.head_dim": [
            [80, 8, 128]
        ],
        "rope_offset.rope_dim": [
            [0, 128]
        ],
        "batch_size.cache_len.q_len": [
            [16, 1024, 1],
            [16, 2048, 1],
            [16, 4096, 1],
            [16, 8192, 1],
            [16, 10240, 1],
            [16, 16384, 1],
            [16, 32768, 1],
            [16, 1024, 4],
            [16, 4096, 4],
            [16, 8192, 4],
            [16, 10240, 4],
            [16, 16384, 4],
            [16, 32768, 4]
        ]
    }
]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def iter_cases(attn_mode):
    """Expand config entries of the given attn_mode into flat test cases."""
    for cfg in ROTARY_EMBEDDING_CONFIGS:
        if cfg["attn_mode"] != attn_mode:
            continue
        dtype = DTYPE_MAP[cfg["dtype"]]
        for (qh, kvh, hd), (ro, rd), (bs, cl, ql) in itertools.product(
                cfg["q_head_num.kv_head_num.head_dim"],
                cfg["rope_offset.rope_dim"],
                cfg["batch_size.cache_len.q_len"]):
            yield dict(q_head_num=qh, kv_head_num=kvh, head_dim=hd,
                       rope_offset=ro, rope_dim=rd,
                       batch_size=bs, cache_len=cl, q_len=ql, dtype=dtype,
                       test_name=f"{attn_mode}_qh{qh}_kvh{kvh}_hd{hd}_ro{ro}_rd{rd}"
                                 f"_bs{bs}_cl{cl}_ql{ql}_{cfg['dtype']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = torch.sum(a_flat * b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    norm_a = norm_a if norm_a.item() > 1e-8 else torch.tensor(1.0, device=a.device)
    norm_b = norm_b if norm_b.item() > 1e-8 else torch.tensor(1.0, device=b.device)
    return (dot / (norm_a * norm_b)).item()


def build_rope_table(cache_len, rope_dim, head_dim, device):
    """Neox-style RoPE cos/sin tables; kernel only reads first ROPE_DIM/2 per row."""
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / head_dim))
    pos = torch.arange(cache_len, dtype=torch.float32)
    angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.zeros(cache_len, rope_dim, dtype=torch.float32)
    sin = torch.zeros(cache_len, rope_dim, dtype=torch.float32)
    cos[:, :half] = torch.cos(angles)
    sin[:, :half] = torch.sin(angles)
    return cos.to(device).contiguous(), sin.to(device).contiguous()


def reference_rotary_embedding(packed_qkv, accum_q_lens, cache_lens, cos, sin,
                               q_head_num, kv_head_num, rope_offset):
    num_tokens, total_head_num, head_dim = packed_qkv.shape
    rope_dim = cos.size(-1)
    half = rope_dim // 2

    out = packed_qkv.clone().float()
    accum = accum_q_lens.to(device=packed_qkv.device)
    cache = cache_lens.to(device=packed_qkv.device)
    slice_idx = torch.arange(num_tokens, device=packed_qkv.device, dtype=accum.dtype)
    bins = torch.searchsorted(accum, slice_idx, right=True) - 1
    cache_idx = (cache[bins] + (slice_idx - accum[bins])).long()

    c = cos[cache_idx][:, :half].unsqueeze(1)
    s = sin[cache_idx][:, :half].unsqueeze(1)

    x = out[:, :, rope_offset:rope_offset + half]
    y = out[:, :, rope_offset + half:rope_offset + rope_dim]
    new_x = x * c - y * s
    new_y = y * c + x * s
    out[:, :, rope_offset:rope_offset + half] = new_x
    out[:, :, rope_offset + half:rope_offset + rope_dim] = new_y
    return out.to(packed_qkv.dtype)


def run_rotary_case(q_head_num, kv_head_num, head_dim, rope_dim, rope_offset,
                    batch_size, cache_len, q_len, test_name, dtype=torch.bfloat16):
    set_seed(42)
    device = 0
    torch.cuda.set_device(device)

    total_head_num = q_head_num + kv_head_num
    num_tokens = batch_size * q_len  # 修正：总 token 数 = batch_size * q_len
    elem_size = torch.tensor([], dtype=dtype).element_size()

    packed_qkv = (torch.randn(num_tokens, total_head_num, head_dim, dtype=dtype) * 0.5).cuda()
    accum_q_lens = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32).cuda()
    cache_lens_t = torch.full((batch_size,), cache_len, dtype=torch.int32).cuda()
    cos, sin = build_rope_table(cache_len + q_len, rope_dim, head_dim, 'cuda')

    golden = reference_rotary_embedding(
        packed_qkv, accum_q_lens, cache_lens_t, cos, sin,
        q_head_num, kv_head_num, rope_offset)

    def kernel():
        rotary_embedding(packed_qkv, accum_q_lens, accum_q_lens, cache_lens_t,
                         cos, sin, q_head_num, kv_head_num, rope_offset)

    kernel()
    torch.cuda.synchronize()

    sim = cosine_similarity(packed_qkv, golden)

    byte_count = 2 * num_tokens * total_head_num * head_dim * elem_size
    flop_count = num_tokens * total_head_num * rope_dim * 3
    arithmetic_intensity = flop_count / byte_count

    stats = measure_cuda(kernel, iters=200, warmup=20, device=device)
    time_us = stats['mean_us']
    bw_gbps = byte_count / (time_us * 1e-6) / 1e9
    flop_gflops = flop_count / (time_us * 1e-6) / 1e9

    print(f"[{test_name}] tokens={num_tokens} total_heads={total_head_num} "
          f"head_dim={head_dim} rope_dim={rope_dim} rope_offset={rope_offset} "
          f"batch={batch_size} cache_len={cache_len} q_len={q_len} dtype={dtype}")
    print(f"  mean={time_us:.3f}us median={stats['median_us']:.3f}us "
          f"min={stats['min_us']:.3f}us stdev={stats['stdev_us']:.3f}us")
    print(f"  bandwidth={bw_gbps:.2f} GB/s  (bytes moved={byte_count})")
    print(f"  arithmetic_intensity={arithmetic_intensity:.4f} FLOP/B  "
          f"(FLOPs={flop_count}, throughput={flop_gflops:.2f} GFLOPS)")
    print(f"  cosine_similarity={sim:.10f}  diff_from_1={abs(1.0 - sim):.2e}")
    assert sim >= 0.9999, f"{test_name} cosine similarity {sim:.6f} < 0.9999 FAILED"
    print(f"  PASS {test_name}\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRotaryEmbedding(unittest.TestCase):

    def _run_mode(self, attn_mode):
        cases = list(iter_cases(attn_mode))
        self.assertTrue(cases, f"no {attn_mode} configs found")
        for case in cases:
            with self.subTest(case["test_name"]):
                run_rotary_case(**case)

    def test_prefill(self):
        self._run_mode("prefill")

    def test_decode(self):
        self._run_mode("decode")


if __name__ == '__main__':
    unittest.main()