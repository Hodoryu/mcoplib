
import pytest
import torch
import torch.nn.functional as F
import mcoplib._C


DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]
NO_MUL_ACTIVATIONS = ["silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul", "relu2_no_mul"]
GATED_ACTIVATIONS = ["silu", "gelu", "gelu_tanh"]
CLAMP_LIMITS = [1.0, 3.0, 7.0]


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if not hasattr(torch.ops._C, "masked_moe_activation"):
        pytest.skip("mcoplib operator masked_moe_activation is not registered")
    return torch.ops._C.masked_moe_activation


def _run_op(output, input, valid_token_counts, activation, clamp_limit=-1.0, alpha=1.0, beta=0.0, situ_beta=1.0, situ_linear_beta=0.0):
    op = _require_op()
    op(output, input, valid_token_counts, activation, clamp_limit, alpha, beta, situ_beta, situ_linear_beta)
    torch.cuda.synchronize()


def _reference_no_mul(input, activation):
    x = input.float()
    if activation == "silu_no_mul":
        output = F.silu(x)
    elif activation == "gelu_no_mul":
        output = F.gelu(x, approximate="none")
    elif activation == "gelu_tanh_no_mul":
        output = F.gelu(x, approximate="tanh")
    elif activation == "relu2_no_mul":
        output = F.relu(x).square()
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return output.to(input.dtype)


def _reference_gated(input, activation):
    gate, up = input.float().chunk(2, dim=-1)
    if activation == "silu":
        activated = F.silu(gate)
    elif activation == "gelu":
        activated = F.gelu(gate, approximate="none")
    elif activation == "gelu_tanh":
        activated = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return (activated * up).to(input.dtype)


def _reference_silu_with_clamp(input, clamp_limit, alpha=1.0, beta=0.0):
    gate, up = input.float().chunk(2, dim=-1)
    gate = gate.clamp(max=clamp_limit)
    up = up.clamp(min=-clamp_limit, max=clamp_limit)
    activated = gate * torch.sigmoid(alpha * gate)
    return (activated * (up + beta)).to(input.dtype)


def _reference_swiglustep(input, clamp_limit):
    gate, up = input.float().chunk(2, dim=-1)
    gate_silu = F.silu(gate).clamp(max=clamp_limit)
    up_clamped = up.clamp(min=-clamp_limit, max=clamp_limit)
    return (gate_silu * up_clamped).to(input.dtype)


def _reference_situ(input, situ_beta, situ_linear_beta):
    gate, up = input.float().chunk(2, dim=-1)
    activated = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
    if situ_linear_beta > 0:
        up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
    return (activated * up).to(input.dtype)


def _check_valid_and_padding(output, expected, valid_count, sentinel=42.0):
    if valid_count > 0:
        torch.testing.assert_close(output[:valid_count], expected, atol=2e-2, rtol=2e-2)
    assert torch.all(output[valid_count:] == sentinel), "Invalid rows were modified"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", GATED_ACTIVATIONS)
@pytest.mark.parametrize("d", [32, 64, 513])
@pytest.mark.parametrize("num_tokens", [1, 7, 67])
@torch.inference_mode()
def test_masked_moe_activation_flat_gated(dtype, activation, d, num_tokens):
    torch.manual_seed(1234)
    input = torch.randn(num_tokens, 2 * d, dtype=dtype, device=DEVICE)
    valid_count = num_tokens // 2
    valid_token_counts = torch.tensor([valid_count], dtype=torch.int32, device=DEVICE)
    output = torch.full((num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, activation)

    expected = _reference_gated(input[:valid_count], activation)
    _check_valid_and_padding(output, expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@pytest.mark.parametrize("d", [32, 64, 513])
@pytest.mark.parametrize("num_tokens", [1, 7, 67])
@torch.inference_mode()
def test_masked_moe_activation_flat_no_mul(dtype, activation, d, num_tokens):
    torch.manual_seed(1234)
    input = torch.randn(num_tokens, d, dtype=dtype, device=DEVICE)
    valid_count = num_tokens // 2
    valid_token_counts = torch.tensor([valid_count], dtype=torch.int32, device=DEVICE)
    output = torch.full((num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, activation)

    expected = _reference_no_mul(input[:valid_count], activation)
    _check_valid_and_padding(output, expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", GATED_ACTIVATIONS)
@pytest.mark.parametrize("d", [32, 64, 513])
@torch.inference_mode()
def test_masked_moe_activation_batched_gated(dtype, activation, d):
    torch.manual_seed(5678)
    num_experts, max_num_tokens = 4, 7
    input = torch.randn(num_experts, max_num_tokens, 2 * d, dtype=dtype, device=DEVICE)
    counts = [0, 1, max_num_tokens // 2, max_num_tokens]
    valid_token_counts = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    output = torch.full((num_experts, max_num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, activation)

    for expert, valid_count in enumerate(counts):
        expected = _reference_gated(input[expert, :valid_count], activation)
        _check_valid_and_padding(output[expert], expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@pytest.mark.parametrize("d", [32, 64, 513])
@torch.inference_mode()
def test_masked_moe_activation_batched_no_mul(dtype, activation, d):
    torch.manual_seed(5678)
    num_experts, max_num_tokens = 4, 7
    input = torch.randn(num_experts, max_num_tokens, d, dtype=dtype, device=DEVICE)
    counts = [0, 1, max_num_tokens // 2, max_num_tokens]
    valid_token_counts = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    output = torch.full((num_experts, max_num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, activation)

    for expert, valid_count in enumerate(counts):
        expected = _reference_no_mul(input[expert, :valid_count], activation)
        _check_valid_and_padding(output[expert], expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clamp_limit", CLAMP_LIMITS)
@pytest.mark.parametrize("d", [32, 64, 513])
@torch.inference_mode()
def test_masked_moe_activation_silu_with_clamp(dtype, clamp_limit, d):
    torch.manual_seed(9012)
    num_experts, max_num_tokens = 4, 7
    input = torch.randn(num_experts, max_num_tokens, 2 * d, dtype=dtype, device=DEVICE) * 10
    counts = [0, 1, 4, 7]
    valid_token_counts = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    output = torch.full((num_experts, max_num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, "silu_with_clamp", clamp_limit=clamp_limit)

    for expert, valid_count in enumerate(counts):
        expected = _reference_silu_with_clamp(input[expert, :valid_count], clamp_limit)
        _check_valid_and_padding(output[expert], expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clamp_limit", CLAMP_LIMITS)
@pytest.mark.parametrize("d", [32, 64, 513])
@torch.inference_mode()
def test_masked_moe_activation_swiglustep(dtype, clamp_limit, d):
    torch.manual_seed(3456)
    num_experts, max_num_tokens = 4, 7
    input = torch.randn(num_experts, max_num_tokens, 2 * d, dtype=dtype, device=DEVICE) * 10
    counts = [0, 1, 4, 7]
    valid_token_counts = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    output = torch.full((num_experts, max_num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, "swiglustep", clamp_limit=clamp_limit)

    for expert, valid_count in enumerate(counts):
        expected = _reference_swiglustep(input[expert, :valid_count], clamp_limit)
        _check_valid_and_padding(output[expert], expected, valid_count)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("situ_beta", [1.0, 1.5])
@pytest.mark.parametrize("situ_linear_beta", [0.0, 2.0])
@torch.inference_mode()
def test_masked_moe_activation_situ(dtype, situ_beta, situ_linear_beta):
    torch.manual_seed(7890)
    num_experts, max_num_tokens, d = 4, 7, 64
    input = torch.randn(num_experts, max_num_tokens, 2 * d, dtype=dtype, device=DEVICE)
    counts = [0, 1, 4, 7]
    valid_token_counts = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    output = torch.full((num_experts, max_num_tokens, d), 42.0, dtype=dtype, device=DEVICE)

    _run_op(output, input, valid_token_counts, "situ", situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)

    for expert, valid_count in enumerate(counts):
        expected = _reference_situ(input[expert, :valid_count], situ_beta, situ_linear_beta)
        _check_valid_and_padding(output[expert], expected, valid_count)