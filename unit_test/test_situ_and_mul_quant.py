# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for mcoplib situ_and_mul_quant."""

import pytest
import torch

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "situ_and_mul_quant"

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
SITU_D = 3072
GROUP_SIZE = 128

if hasattr(torch, "float8_e4m3fn"):
    FP8_DTYPE = torch.float8_e4m3fn
elif hasattr(torch, "float8_e4m3fnuz"):
    FP8_DTYPE = torch.float8_e4m3fnuz
else:
    FP8_DTYPE = None

FP8_MAX = torch.finfo(FP8_DTYPE).max if FP8_DTYPE is not None else None

CASES = [
    (1, None),
    (17, None),
    (128, None),
    (2048, None),
    (4096, None),
    (2048, 1500),
    (2048, 1),
    (4096, 4095),
    (512, 0),
]


def _get_op():
    """Resolve the registered mcoplib operator."""
    if FP8_DTYPE is None:
        pytest.skip("PyTorch FP8 dtype is unavailable")

    for namespace_name in ("_C", "mcoplib"):
        namespace = getattr(torch.ops, namespace_name, None)
        if namespace is not None and hasattr(namespace, OP_NAME):
            return getattr(namespace, OP_NAME)

    pytest.skip(
        f"Operator {OP_NAME!r} is not registered in torch.ops._C "
        "or torch.ops.mcoplib."
    )


def _require_device():
    if not torch.cuda.is_available():
        pytest.skip("No accelerator available through torch.cuda")

    if FP8_DTYPE is None:
        pytest.skip("PyTorch FP8 dtype is unavailable")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_random(
    num_tokens: int,
    twod: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate inputs with a wide range of row magnitudes."""
    base = torch.randn(
        num_tokens,
        twod,
        dtype=torch.float32,
        device=DEVICE,
    )
    exps = torch.randint(
        1,
        13,
        (num_tokens, 1),
        device=DEVICE,
    ).float()
    return (base * (2.0**exps)).to(dtype)


def ref_situ_block_fp8_quant(
    inp: torch.Tensor,
    beta: float,
    linear_beta: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an FP64 SITU reference and ideal block-FP8 quantization."""
    d = inp.shape[-1] // 2
    gate, up = inp.double().chunk(2, dim=-1)

    gate_out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_out = (
        linear_beta * torch.tanh(up / linear_beta)
        if linear_beta > 0
        else up
    )

    # Match the activation cast used before calculating group absmax.
    act = (gate_out * up_out).to(inp.dtype).float()

    m = act.shape[0]
    ng = d // group_size
    a = act.reshape(m, ng, group_size)

    scale = a.abs().amax(dim=2).clamp_min(1e-30) / FP8_MAX
    q = (
        (a / scale.unsqueeze(2))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8_DTYPE)
    )

    return q.reshape(m, d), scale


def fp8_code_dist(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute absolute FP8 encoding distance."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.contiguous().view(torch.uint8).to(torch.int32)
        magnitude = bits & 0x7F
        return torch.where(
            (bits & 0x80) != 0,
            -magnitude,
            magnitude,
        )

    return (ordered(a) - ordered(b)).abs()


def _run_op(
    out: torch.Tensor,
    scale: torch.Tensor,
    inp: torch.Tensor,
    valid_rows: torch.Tensor | None = None,
    topk: int = 1,
) -> None:
    """Call the mcoplib operator."""
    op = _get_op()

    op(
        out,
        scale,
        inp,
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        group_size=GROUP_SIZE,
        valid_rows=valid_rows,
        topk=topk,
    )


def _make_outputs(
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.full(
        (num_tokens, SITU_D),
        17.0,
        device=DEVICE,
        dtype=FP8_DTYPE,
    )
    scale = torch.full(
        (num_tokens, SITU_D // GROUP_SIZE),
        -1.0,
        device=DEVICE,
        dtype=torch.float32,
    )
    return out, scale, out.clone()


@pytest.mark.parametrize("num_tokens,valid", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_situ_and_mul_quant(
    num_tokens: int,
    valid: int | None,
    dtype: torch.dtype,
) -> None:
    _require_device()
    _set_seed(42)

    inp = token_random(num_tokens, 2 * SITU_D, dtype)
    out, scale, out_sentinel = _make_outputs(num_tokens)

    valid_rows = (
        None
        if valid is None
        else torch.tensor(valid, dtype=torch.int32, device=DEVICE)
    )

    _run_op(
        out,
        scale,
        inp,
        valid_rows=valid_rows,
    )

    # Padding rows retain their output bytes and receive scale 1.0.
    if valid is not None and valid < num_tokens:
        assert torch.equal(
            scale[valid:],
            torch.ones_like(scale[valid:]),
        )
        assert torch.equal(
            out[valid:].contiguous().view(torch.uint8),
            out_sentinel[valid:].contiguous().view(torch.uint8),
        )

    rows = num_tokens if valid is None else valid
    if rows == 0:
        return

    ref_q, ref_s = ref_situ_block_fp8_quant(
        inp[:rows],
        SITU_BETA,
        SITU_LINEAR_BETA,
        GROUP_SIZE,
    )

    assert torch.isfinite(scale[:rows]).all()
    assert torch.isfinite(out[:rows].float()).all()

    torch.testing.assert_close(
        scale[:rows],
        ref_s,
        rtol=3e-2,
        atol=1e-5,
    )

    dist = fp8_code_dist(out[:rows], ref_q)
    assert dist.max().item() <= 2
    assert (dist <= 1).float().mean().item() >= 0.99


@pytest.mark.parametrize(
    "tokens,topk",
    [
        (300, 8),
        (1, 4),
        (256, 6),
    ],
)
@torch.inference_mode()
def test_situ_and_mul_quant_topk_row_bound(
    tokens: int,
    topk: int,
) -> None:
    """valid_rows is a token count; actual valid rows equal tokens * topk."""
    _require_device()
    _set_seed(7)

    rows = tokens * topk
    num_tokens = rows + 37

    inp = token_random(
        num_tokens,
        2 * SITU_D,
        torch.bfloat16,
    )
    out, scale, out_sentinel = _make_outputs(num_tokens)

    valid_rows = torch.tensor(
        tokens,
        dtype=torch.int32,
        device=DEVICE,
    )

    _run_op(
        out,
        scale,
        inp,
        valid_rows=valid_rows,
        topk=topk,
    )

    assert torch.equal(
        scale[rows:],
        torch.ones_like(scale[rows:]),
    )
    assert torch.equal(
        out[rows:].contiguous().view(torch.uint8),
        out_sentinel[rows:].contiguous().view(torch.uint8),
    )

    ref_q, ref_s = ref_situ_block_fp8_quant(
        inp[:rows],
        SITU_BETA,
        SITU_LINEAR_BETA,
        GROUP_SIZE,
    )

    torch.testing.assert_close(
        scale[:rows],
        ref_s,
        rtol=3e-2,
        atol=1e-5,
    )

    assert fp8_code_dist(out[:rows], ref_q).max().item() <= 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_situ_and_mul_quant_deterministic(
    dtype: torch.dtype,
) -> None:
    """Repeated runs on the same input must be byte-identical."""
    _require_device()
    _set_seed(0)

    num_tokens = 2048
    inp = token_random(num_tokens, 2 * SITU_D, dtype)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty(
            (num_tokens, SITU_D),
            dtype=FP8_DTYPE,
            device=DEVICE,
        )
        scale = torch.empty(
            (num_tokens, SITU_D // GROUP_SIZE),
            dtype=torch.float32,
            device=DEVICE,
        )

        _run_op(out, scale, inp)
        return out, scale

    out1, scale1 = run()
    out2, scale2 = run()

    assert torch.equal(
        out1.contiguous().view(torch.uint8),
        out2.contiguous().view(torch.uint8),
    )
    assert torch.equal(scale1, scale2)