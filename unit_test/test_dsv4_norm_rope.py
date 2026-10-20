"""Tests for DeepSeek-V4 fused norm + RoPE kernels."""

import math

import pytest

import torch
import mcoplib.sgl_kernel


def _ref_rmsnorm_self(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference: RMSNorm without weight (identity weight)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


def _ref_rope_interleaved(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """Reference: apply RoPE to the last `rope_dim` elements (interleaved re/im)."""
    out = x.clone()
    B = x.size(0)
    head_dim = x.size(-1)
    nope_dim = head_dim - rope_dim

    for b in range(B):
        pos = positions[b].item()
        freq = freqs_cis[pos]  # (rope_dim,) interleaved [re0, im0, re1, im1, ...]
        rope_part = out[b, ..., nope_dim:].float()
        # Reshape to pairs
        pairs = rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        x_real = pairs[..., 0]
        x_imag = pairs[..., 1]
        freq_pairs = freq.reshape(rope_dim // 2, 2)
        f_real = freq_pairs[:, 0]
        f_imag = freq_pairs[:, 1]
        rot_real = x_real * f_real - x_imag * f_imag
        rot_imag = x_real * f_imag + x_imag * f_real
        result = torch.stack([rot_real, rot_imag], dim=-1).reshape(rope_part.shape)
        out[b, ..., nope_dim:] = result.to(x.dtype)
    return out


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_fused_q_norm_rope_correctness(batch_size, num_heads, head_dim):
    """Test Q norm + rope against reference."""
    torch.manual_seed(42)
    rope_dim = 64
    max_pos = 512
    eps = 1e-6

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="cuda"
    )

    q_output = torch.empty_like(q_input)
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(q_input, q_output, freqs_cis, positions, eps)

    # Reference
    normed = _ref_rmsnorm_self(q_input, eps)
    expected = _ref_rope_interleaved(normed, freqs_cis, positions, rope_dim)

    torch.testing.assert_close(q_output.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fused_q_norm_rope_zero_batch():
    """Empty batch should not crash."""
    q_input = torch.empty(0, 8, 192, dtype=torch.bfloat16, device="cuda")
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device="cuda")
    positions = torch.empty(0, dtype=torch.int32, device="cuda")
    q_output = torch.empty_like(q_input)
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(q_input, q_output, freqs_cis, positions, 1e-6)
    assert q_output.shape == q_input.shape


def test_fused_q_norm_rope_preallocated_output():
    """Test with pre-allocated output tensor."""
    torch.manual_seed(42)
    B, H, D = 4, 8, 192
    q_input = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 512, (B,), dtype=torch.int32, device="cuda")
    q_output = torch.empty_like(q_input)

    # The op writes into q_output in-place and returns nothing.
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(
        q_input, q_output, freqs_cis, positions, 1e-6
    )
    # Verify the pre-allocated tensor was actually written to.
    normed = _ref_rmsnorm_self(q_input, 1e-6)
    expected = _ref_rope_interleaved(normed, freqs_cis, positions, 64)
    torch.testing.assert_close(q_output.float(), expected.float(), rtol=1e-2, atol=1e-2)


# ======================= head_dim = 512 新增用例 =======================
D512_HEAD_DIM = 512
D512_ROPE_DIM = 64
D512_MAX_POS = 65536
D512_EPS = 1e-6
D512_TOL = 1.6e-2


def _make_freqs_cis_d512(max_pos=D512_MAX_POS, rope_dim=D512_ROPE_DIM, theta=10000.0):
    """真实 RoPE 表，交错存储 [cos0, sin0, cos1, sin1, ...]。"""
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return (
        torch.stack([ang.cos(), ang.sin()], -1)
        .reshape(max_pos, rope_dim)
        .contiguous()
        .to("cuda")
    )


def _ref_q_norm_rope_d512(q, freqs_cis, positions, eps=D512_EPS, match_kernel_rounding=True):
    """向量化参考实现；kernel 在 norm 后先舍入为 bf16。"""
    nope_dim = D512_HEAD_DIM - D512_ROPE_DIM
    x = q.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if match_kernel_rounding:
        y = y.to(q.dtype).float()
    nope = y[..., :nope_dim]
    rope = y[..., nope_dim:].reshape(*y.shape[:-1], D512_ROPE_DIM // 2, 2)
    f = freqs_cis[positions.long()].reshape(-1, 1, D512_ROPE_DIM // 2, 2)
    xr, xi, fr, fi = rope[..., 0], rope[..., 1], f[..., 0], f[..., 1]
    rot = torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], -1).reshape(
        *y.shape[:-1], D512_ROPE_DIM
    )
    return torch.cat([nope, rot], -1).to(q.dtype)


def _make_q_d512(B, H, noncontig_batch=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if noncontig_batch:  # batch 维带 padding：stride(0) != H*D
        buf = torch.randn(
            B, H * D512_HEAD_DIM + 256, device="cuda", dtype=torch.bfloat16, generator=g
        )
        return buf[:, : H * D512_HEAD_DIM].view(B, H, D512_HEAD_DIM)
    return torch.randn(B, H, D512_HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)


def _run_and_check_d512(q, freqs_cis, positions):
    q_backup = q.clone()
    q_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(q, q_output, freqs_cis, positions, D512_EPS)
    torch.cuda.synchronize()

    expected = _ref_q_norm_rope_d512(q, freqs_cis, positions)
    expected32 = _ref_q_norm_rope_d512(q, freqs_cis, positions, match_kernel_rounding=False)

    assert not torch.isnan(q_output).any(), "output contains NaN"
    assert torch.equal(q, q_backup), "kernel modified the input tensor"
    torch.testing.assert_close(
        q_output.float(), expected.float(), rtol=D512_TOL, atol=D512_TOL
    )
    cos = torch.nn.functional.cosine_similarity(
        q_output.float().flatten(), expected32.float().flatten(), dim=0
    ).item()
    assert cos > 0.9999, f"cosine similarity too low: {cos}"


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 3, 7, 33, 257, 1024])
def test_fused_q_norm_rope_d512_correctness(batch_size, num_heads, pos_dtype):
    q = _make_q_d512(batch_size, num_heads)
    freqs_cis = _make_freqs_cis_d512()
    g = torch.Generator(device="cuda").manual_seed(0)
    positions = torch.randint(
        0, D512_MAX_POS, (batch_size,), dtype=pos_dtype, device="cuda", generator=g
    )
    _run_and_check_d512(q, freqs_cis, positions)


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("num_heads", [64, 128])
def test_fused_q_norm_rope_d512_noncontig_batch(num_heads, pos_dtype):
    B = 17
    q = _make_q_d512(B, num_heads, noncontig_batch=True)
    assert not q.is_contiguous()
    freqs_cis = _make_freqs_cis_d512()
    positions = torch.randint(0, D512_MAX_POS, (B,), dtype=pos_dtype, device="cuda")
    _run_and_check_d512(q, freqs_cis, positions)


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_fused_q_norm_rope_d512_edge_positions(pos_dtype):
    q = _make_q_d512(8, 16)
    freqs_cis = _make_freqs_cis_d512()
    positions = torch.tensor([0, D512_MAX_POS - 1] * 4, device="cuda", dtype=pos_dtype)
    _run_and_check_d512(q, freqs_cis, positions)


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_fused_q_norm_rope_d512_zero_batch(pos_dtype):
    q_input = torch.empty(0, 64, D512_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    freqs_cis = _make_freqs_cis_d512()
    positions = torch.empty(0, dtype=pos_dtype, device="cuda")
    q_output = torch.empty_like(q_input)
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(
        q_input, q_output, freqs_cis, positions, D512_EPS
    )
    assert q_output.shape == q_input.shape


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_q_indexer_rope_hadamard_quant_runs(batch_size):
    """Smoke test: kernel runs without errors and produces finite results."""
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 128
    rope_dim = 64
    max_pos = 256

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8 = torch.empty(
        batch_size, num_heads, head_dim, dtype=torch.uint8, device="cuda"
    )
    weight = torch.randn(batch_size, num_heads, dtype=torch.bfloat16, device="cuda")
    weights_out = torch.empty(
        batch_size, num_heads, 1, dtype=torch.float32, device="cuda"
    )
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="cuda"
    )
    weight_scale = 0.5

    torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
        q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions
    )

    assert torch.isfinite(weights_out).all(), "weights_out contains non-finite values"
    assert q_fp8.any(), "q_fp8 should not be all zeros"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))