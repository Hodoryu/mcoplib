# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "fused_gdn_decode_post_conv_mtp"

K = 128
V = 128
RATIO = 8
SUPPORTED_RATIOS = (1, 2, 3, 4, 8)
SUPPORTED_ACTIVATIONS = ("silu", "sigmoid")


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _require_sm80():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("fused GDN decode MTP requires compute capability 8.0+")


def _rmsnorm_reference(x, weight, z=None, eps=1e-6, output_gate_activation="silu"):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    x_norm = x_norm * weight.float()
    if z is not None:
        z_float = z.float()
        if output_gate_activation == "silu":
            gate = F.silu(z_float)
        elif output_gate_activation == "sigmoid":
            gate = torch.sigmoid(z_float)
        else:
            raise ValueError(f"Unsupported output_gate_activation: {output_gate_activation}")
        x_norm = x_norm * gate
    return x_norm.to(x.dtype)


def _gdn_update_reference(A_log, a, b, dt_bias, q, k, v, initial_state, state_indices, cu_seqlens, num_accepted_tokens, scale):
    q = q.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    A_log = A_log.float()
    dt_bias = dt_bias.float()

    num_tokens, num_heads, dim_k = q.shape
    num_kv_heads = v.shape[1]
    dim_v = v.shape[-1]

    assert num_kv_heads % num_heads == 0
    assert num_kv_heads // num_heads in SUPPORTED_RATIOS
    assert dim_k == K
    assert dim_v == V

    repeat_ratio = num_kv_heads // num_heads

    q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1.0e-6)
    k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1.0e-6)
    q = q * scale

    x = a + dt_bias.view(1, num_kv_heads)
    beta_sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    g = -torch.exp(A_log).view(1, num_kv_heads) * beta_sp
    decay = torch.exp(g)
    beta = torch.sigmoid(b)

    state_dtype = initial_state.dtype
    state = initial_state.float().clone()
    output = torch.zeros((num_tokens, num_kv_heads, dim_v), dtype=torch.float32, device=q.device)

    num_requests = state_indices.shape[0]
    state_indices_width = state_indices.shape[1]

    for req_idx in range(num_requests):
        bos = int(cu_seqlens[req_idx].item())
        eos = int(cu_seqlens[req_idx + 1].item())
        req_tokens = eos - bos

        if req_tokens <= 0:
            continue

        accepted = int(num_accepted_tokens[req_idx].item())

        if accepted <= 0 or accepted > state_indices_width:
            output[bos:eos].zero_()
            continue

        source_slot = int(state_indices[req_idx, accepted - 1].item())

        if source_slot <= 0 or req_tokens > 8:
            output[bos:eos].zero_()
            continue

        for value_head in range(num_kv_heads):
            key_head = value_head // repeat_ratio
            h = state[source_slot, value_head].clone()

            for t in range(req_tokens):
                token = bos + t
                q_vec = q[token, key_head]
                k_vec = k[token, key_head]
                v_vec = v[token, value_head]

                h = h * decay[token, value_head]
                state_k = torch.mv(h, k_vec)
                delta = (v_vec - state_k) * beta[token, value_head]
                h = h + torch.outer(delta, k_vec)
                output[token, value_head] = torch.mv(h, q_vec)

                if t < state_indices_width:
                    destination_slot = int(state_indices[req_idx, t].item())
                    if destination_slot > 0:
                        if state_dtype == torch.bfloat16:
                            state[destination_slot, value_head].copy_(h.to(torch.bfloat16).float())
                        else:
                            state[destination_slot, value_head].copy_(h)

    return output, state.to(state_dtype)


def _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation="silu"):
    num_tokens = mixed_qkv.shape[0]
    query, key, value = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    query = query.view(num_tokens, H, K)
    key = key.view(num_tokens, H, K)
    value = value.view(num_tokens, HV, V)

    raw_ref, state_out = _gdn_update_reference(A_log, a, b, dt_bias, query, key, value, state_ref, state_indices, cu_seqlens, num_accepted_tokens, scale)
    raw_ref = raw_ref.to(torch.bfloat16)
    expected = _rmsnorm_reference(raw_ref, norm_weight, z=output_gate, eps=eps, output_gate_activation=output_gate_activation)
    return expected, state_out


def _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate, norm_weight, scale, norm_eps, output_gate_activation="silu"):
    assert mixed_qkv.is_cuda and mixed_qkv.dtype == torch.bfloat16
    assert a.is_cuda and a.dtype == torch.bfloat16 and a.stride(1) == 1
    assert b.is_cuda and b.dtype == torch.bfloat16 and b.stride(1) == 1
    assert A_log.is_cuda and A_log.dtype == torch.float32
    assert dt_bias.is_cuda and dt_bias.dtype in (torch.float32, torch.bfloat16, torch.float16)
    assert state.is_cuda and state.dtype in (torch.float32, torch.bfloat16)
    assert output_gate.is_cuda and output_gate.dtype == torch.bfloat16
    assert norm_weight.is_cuda and norm_weight.dtype in (torch.float32, torch.bfloat16)

    out = torch.zeros_like(output_gate)
    result = torch.ops._C.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=state_indices, cu_seqlens=cu_seqlens, num_accepted_tokens=num_accepted_tokens, state=state, output_gate=output_gate, norm_weight=norm_weight, out=out, scale=scale, norm_eps=norm_eps, output_gate_activation=output_gate_activation)

    assert result is None
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    return out


def _make_inputs(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype=torch.float32, state_width=None, seed=0, ratio=RATIO):
    torch.manual_seed(seed)

    H = tp_size
    HV = H * ratio
    num_tokens = sum(query_lengths)
    num_requests = len(query_lengths)

    assert num_tokens > 0
    assert ratio in SUPPORTED_RATIOS

    max_query_length = max(query_lengths)
    if state_width is None:
        state_width = max_query_length

    assert state_width > 0 and state_width <= 8

    num_slots = max(32, num_requests * max(max_query_length, state_width) + 8)

    mixed_qkv = torch.randn((num_tokens, 2 * H * K + HV * V), dtype=torch.bfloat16, device=DEVICE)
    ba = torch.randn((num_tokens, 2 * HV), dtype=torch.bfloat16, device=DEVICE)
    b, a = ba.chunk(2, dim=-1)
    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(HV, dtype=dt_bias_dtype, device=DEVICE)
    output_gate = torch.randn((num_tokens, HV, V), dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=norm_dtype, device=DEVICE)
    state_ref = (0.01 * torch.randn((num_slots, HV, V, K), dtype=torch.float32, device=DEVICE)).to(state_dtype)
    state_actual = state_ref.clone()

    state_indices = torch.zeros((num_requests, state_width), dtype=torch.int32, device=DEVICE)
    next_slot = 1
    for req_idx, req_len in enumerate(query_lengths):
        for t in range(min(req_len, state_width)):
            state_indices[req_idx, t] = next_slot
            next_slot += 1

    cu_seqlens = [0]
    for req_len in query_lengths:
        cu_seqlens.append(cu_seqlens[-1] + req_len)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=DEVICE)
    num_accepted_tokens = torch.ones(num_requests, dtype=torch.int32, device=DEVICE)

    scale = 1.0 / (K ** 0.5)
    eps = 1.0e-6

    return (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)


def _assert_matches_reference(actual, expected, state_actual, state_expected, state_dtype, message=""):
    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)
    assert output_relative_l2 < 5e-4, f"{message} output relative L2 mismatch: {output_relative_l2.item():.6g}"

    if state_dtype == torch.float32:
        state_atol = 5e-5
        state_rtol = 5e-5
    else:
        state_atol = 5e-3
        state_rtol = 5e-3

    assert state_actual.dtype == state_expected.dtype
    torch.testing.assert_close(state_actual, state_expected, rtol=state_rtol, atol=state_atol)


@pytest.mark.parametrize("ratio", SUPPORTED_RATIOS)
@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@pytest.mark.parametrize("state_dtype,norm_dtype,dt_bias_dtype", [
    (torch.bfloat16, torch.bfloat16, torch.float32),
    (torch.float32, torch.float32, torch.bfloat16),
    (torch.float32, torch.bfloat16, torch.float16),
])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_dispatch(ratio, output_gate_activation, state_dtype, norm_dtype, dt_bias_dtype):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4, 2), state_dtype, norm_dtype, dt_bias_dtype=dt_bias_dtype, seed=100 + ratio, ratio=ratio)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    _assert_matches_reference(actual, expected, state_actual, state_expected, state_dtype, message=f"ratio={ratio}, activation={output_gate_activation}")


@pytest.mark.parametrize("tp_size,query_lengths,state_dtype,norm_dtype,dt_bias_dtype", [
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.float32, id="tp16-bf16-dtbias-fp32"),
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.bfloat16, id="tp16-bf16-dtbias-bf16"),
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.float16, id="tp16-bf16-dtbias-fp16"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.float32, id="tp4-fp32-dtbias-fp32"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.bfloat16, id="tp4-fp32-dtbias-bf16"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.float16, id="tp4-fp32-dtbias-fp16"),
    pytest.param(16, (4, 2, 0), torch.bfloat16, torch.float32, torch.float32, id="tp16-ragged-dtbias-fp32"),
    pytest.param(4, (4, 2, 0), torch.float32, torch.bfloat16, torch.bfloat16, id="tp4-ragged-dtbias-bf16"),
    pytest.param(16, (8,), torch.float32, torch.bfloat16, torch.float16, id="tp16-max-dtbias-fp16"),
    pytest.param(4, (8,), torch.bfloat16, torch.float32, torch.float32, id="tp4-max-dtbias-fp32"),
])
@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_ratio8(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype, output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype=dt_bias_dtype, seed=0)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    assert mixed_qkv.dim() == 2 and mixed_qkv.dtype == torch.bfloat16
    assert a.shape == (sum(query_lengths), HV) and b.shape == (sum(query_lengths), HV)
    assert state_actual.shape == (state_ref.shape[0], HV, V, K)

    state_width = state_indices.shape[1]
    for step, accepted_tokens in enumerate((1, min(2, state_width), state_width)):
        num_accepted_tokens.fill_(accepted_tokens)
        if query_lengths[-1] == 0:
            num_accepted_tokens[-1] = 1

        expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
        actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

        _assert_matches_reference(actual, expected, state_actual, state_expected, state_dtype, message=f"step={step}, accepted={accepted_tokens}, activation={output_gate_activation}")
        state_ref = state_expected.clone()


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_output_shape(state_dtype, output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), state_dtype, torch.float32, seed=1)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)
    assert actual.shape == (4, HV, V)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("dt_bias_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_dt_bias_dtype(dt_bias_dtype, output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.float32, torch.float32, dt_bias_dtype=dt_bias_dtype, seed=10)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)
    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"dt_bias={dt_bias_dtype}, activation={output_gate_activation}")


@pytest.mark.parametrize("state_width", [1, 2, 4, 8])
@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_state_indices_width(state_width, output_gate_activation):
    _require_op()
    _require_sm80()

    query_length = min(state_width, 8)
    inputs = _make_inputs(4, (query_length,), torch.float32, torch.float32, state_width=state_width, seed=20 + state_width)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)
    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"state_width={state_width}, activation={output_gate_activation}")


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_accepted_greater_than_width(output_gate_activation):
    _require_op()
    _require_sm80()

    state_width = 4
    inputs = _make_inputs(4, (4,), torch.bfloat16, torch.float32, state_width=state_width, seed=30)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    num_accepted_tokens.fill_(state_width + 1)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_source_slot_zero(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=31)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    state_indices[0, 0] = 0
    num_accepted_tokens.fill_(1)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_zero_accepted(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=4)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    num_accepted_tokens.zero_()
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_num_tokens_eight(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (8,), torch.float32, torch.float32, state_width=8, seed=40)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)
    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"8-token activation={output_gate_activation}")


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_num_tokens_greater_than_eight(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (9,), torch.float32, torch.float32, state_width=8, seed=41)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    num_accepted_tokens.fill_(1)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_no_nan(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (8,), torch.bfloat16, torch.float32, seed=2)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_ragged(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4, 2, 0), torch.float32, torch.float32, seed=3)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    assert cu_seqlens.tolist() == [0, 4, 6, 6]
    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert actual.shape == (6, HV, V)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()
    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"ragged activation={output_gate_activation}")


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_empty_request(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4, 0), torch.float32, torch.float32, seed=50)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    assert cu_seqlens.tolist() == [0, 4, 4]
    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert actual.shape == (4, HV, V)
    assert actual.dtype == torch.bfloat16
    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"empty request activation={output_gate_activation}")


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_state_update(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=5)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    before = state_actual.clone()
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    assert (state_actual != before).any().item()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()


@pytest.mark.parametrize("output_gate_activation", SUPPORTED_ACTIVATIONS)
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_scale_and_norm_eps(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.float32, torch.float32, seed=60)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    scale = 0.25
    eps = 1.0e-5
    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V, output_gate_activation=output_gate_activation)
    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)

    _assert_matches_reference(actual, expected, state_actual, state_expected, torch.float32, message=f"scale/norm_eps activation={output_gate_activation}")


@pytest.mark.parametrize("output_gate_activation", ["", "relu", "SiLU", "SIGMOID", "tanh"])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_invalid_output_gate_activation(output_gate_activation):
    _require_op()
    _require_sm80()

    inputs = _make_inputs(4, (4,), torch.float32, torch.float32, seed=70)
    mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V = inputs

    with pytest.raises(RuntimeError, match="output_gate_activation must be 'silu' or 'sigmoid'"):
        _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps, output_gate_activation=output_gate_activation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])