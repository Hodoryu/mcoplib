"""Native mHC pre big-fuse tests: legacy and hidden=5120 delayed paths.

TileLang implementation tests live separately in test_tilelang_mhc_pre.py.

Accuracy is validated against the vLLM-metax original implementation
(`mhc_pre_delayed_tilelang` -> `MHC_PRE_NORM_KERNEL` /
`mhc_pre_big_fuse_tilelang`), with the independent PyTorch reference kept as a
cross-check.

The vLLM reference lives on the vLLM-metax `master` branch.  Point the test at
a checkout of it either by installing vLLM-metax or by exporting::

    VLLM_METAX_ROOT=/path/to/vLLM-metax-master

Tests that compare against vLLM are skipped when that import is unavailable.

Run::

    pytest -q -s unit_test/test_mhc_pre_big_fuse.py

Run only the DSV4.1-Flash-INT8-TP8DP2 server-log test set::

    pytest -q unit_test/test_mhc_pre_big_fuse.py -k dsv41_flash_int8_tp8dp2

This set uses the supplied summary of
``Server-Metax-TP8DP2-MHCShape-vllm-V1.log``: all 13 observed token counts,
both delayed paths, n_splits=16, fused RMSNorm, saved next pre-mix, and no
aux output (the logged 11-tensor signature). INT8 describes the model;
this operator's inputs remain BF16/FP32. GEMM partials are constructed with
PyTorch; these tests exercise big-fuse, not the upstream GEMM implementation.
"""

import os
import sys

import pytest
import torch

pytest.importorskip("mcoplib.op")

from mcoplib import op as mcoplib_ops  # noqa: E402

HIDDEN = 5120
HC_MULT = 4
MHC_MULT3 = 24
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20
NORM_EPS = 1e-6

# Observed N values, not a synthetic decode/prefill sweep. Keep the server-log
# split count independent of the generic extension tests below.
DSV41_FLASH_INT8_TP8DP2_TOKENS = (
    1, 2, 3, 4, 16, 30, 61, 94, 99, 123, 126, 256, 2048
)
DSV41_FLASH_INT8_TP8DP2_SPLITS = 16
NUM_TOKENS = DSV41_FLASH_INT8_TP8DP2_TOKENS
NUM_SPLITS = (1, 4, 16)


# --------------------------------------------------------------------------
# vLLM-metax original implementation (reference for accuracy).
# --------------------------------------------------------------------------
def _load_vllm_reference():
    root = os.environ.get("VLLM_METAX_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from vllm_metax.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_pre_big_fuse_tilelang,
        )
        from vllm_metax.model_executor.kernels.mhc.warmup import (
            MHC_PRE_NORM_KERNEL,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, exc
    return MHC_PRE_NORM_KERNEL, mhc_pre_big_fuse_tilelang, None


MHC_PRE_NORM_KERNEL, _vllm_mhc_pre_big_fuse_tilelang, _VLLM_IMPORT_ERROR = (
    _load_vllm_reference()
)
VLLM_REFERENCE_AVAILABLE = MHC_PRE_NORM_KERNEL is not None
VLLM_SKIP_REASON = (
    "vLLM-metax reference unavailable (set VLLM_METAX_ROOT to a master "
    f"checkout): {_VLLM_IMPORT_ERROR!r}"
)


def vllm_reference(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    norm_weight,
    pre_mix_in,
    save_pre_mix,
    rms_numel,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_eps,
    write_aux,
):
    """Mirror vLLM-metax ``mhc_pre_delayed_tilelang`` (master: tilelang.py:75).

    The delayed wrapper picks the fused-RMSNorm kernel when ``norm_weight`` is
    given and the plain one otherwise; both receive a stand-in tensor when an
    optional input is unused, exactly like the original.
    """
    num_tokens, mhc_mult, hidden_size = residual.shape
    post = torch.empty(num_tokens, mhc_mult, device="cuda", dtype=torch.float32)
    comb = torch.empty(
        num_tokens, mhc_mult * mhc_mult, device="cuda", dtype=torch.float32
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    next_pre_mix = torch.empty(
        num_tokens, mhc_mult, device="cuda", dtype=torch.float32
    )
    aux_out = torch.empty(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    pre_mix_arg = pre_mix_in if pre_mix_in is not None else post
    use_pre_mix_in = pre_mix_in is not None

    if norm_weight is not None:
        MHC_PRE_NORM_KERNEL(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix_arg,
            next_pre_mix,
            aux_out,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=mhc_mult,
            use_pre_mix_in=use_pre_mix_in,
            save_pre_mix=save_pre_mix,
            rms_numel=rms_numel,
            write_aux=write_aux,
        )
    else:
        _vllm_mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            pre_mix_arg,
            next_pre_mix,
            aux_out,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            gemm_out_mul.shape[0],
            mhc_mult,
            use_pre_mix_in=use_pre_mix_in,
            save_pre_mix=save_pre_mix,
            rms_numel=rms_numel,
            write_aux=write_aux,
        )
    return (
        post,
        comb,
        layer_input,
        next_pre_mix,
        aux_out if write_aux else None,
    )


# --------------------------------------------------------------------------
# Independent PyTorch reference (cross-check only).
# --------------------------------------------------------------------------
def torch_reference(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    norm_weight,
    pre_mix_in,
    save_pre_mix,
    rms_numel,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_eps,
    write_aux,
):
    _, num_tokens, _ = gemm_out_mul.shape
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]

    rms = torch.rsqrt(gemm_out_sqrsum.sum(0) / rms_numel + rms_eps)
    mixes = gemm_out_mul.sum(0) * rms.unsqueeze(-1)

    pre_cur = (
        torch.sigmoid(mixes[:, :mhc_mult] * hc_scale[0] + hc_base[:mhc_mult])
        + hc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, mhc_mult : 2 * mhc_mult] * hc_scale[1]
            + hc_base[mhc_mult : 2 * mhc_mult]
        )
        * hc_post_mult_value
    )
    comb = (
        mixes[:, 2 * mhc_mult :] * hc_scale[2] + hc_base[2 * mhc_mult :]
    ).view(num_tokens, mhc_mult, mhc_mult)
    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    if pre_mix_in is not None:
        active = pre_mix_in
    elif save_pre_mix:
        active = torch.zeros(
            num_tokens, mhc_mult, device=residual.device, dtype=torch.float32
        )
        active[:, 0] = 1.0
    else:
        active = pre_cur

    collapsed = (residual.float() * active.unsqueeze(-1)).sum(1)
    collapsed_bf16 = collapsed.bfloat16()
    if norm_weight is not None:
        base = collapsed_bf16.float() if save_pre_mix else collapsed
        inv_rms = torch.rsqrt(
            (base * base).sum(-1, keepdim=True) / hidden_size + norm_eps
        )
        layer_input = (
            collapsed_bf16.float() * inv_rms * norm_weight.float()
        ).bfloat16()
    else:
        layer_input = collapsed_bf16
    aux = residual.float().mean(1).bfloat16() if write_aux else None
    return (
        post,
        comb.reshape(num_tokens, mhc_mult * mhc_mult),
        layer_input,
        pre_cur,
        aux,
    )


def run_reference(name, *args, **kwargs):
    if name == "vllm":
        return vllm_reference(*args, **kwargs)
    return torch_reference(*args, **kwargs)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------
def make_gemm(x, fn, n_splits):
    num_tokens, input_size = x.shape
    assert input_size % n_splits == 0
    k_chunk = input_size // n_splits
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, MHC_MULT3, device=x.device, dtype=torch.float32
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, device=x.device, dtype=torch.float32
    )
    x_float = x.float()
    for i_split in range(n_splits):
        a = x_float[:, i_split * k_chunk : (i_split + 1) * k_chunk]
        b = fn[:, i_split * k_chunk : (i_split + 1) * k_chunk]
        gemm_out_mul[i_split] = a @ b.t()
        gemm_out_sqrsum[i_split] = a.square().sum(-1)
    return gemm_out_mul, gemm_out_sqrsum


def build_case(num_tokens, n_splits, path, norm, seed):
    torch.manual_seed(seed)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32) * 0.2 + 1.0
    hc_base = torch.randn(MHC_MULT3, device="cuda", dtype=torch.float32) * 0.2
    hidden = HIDDEN
    fn_full = (
        torch.randn(MHC_MULT3, HC_MULT * hidden, device="cuda", dtype=torch.float32)
        / (HC_MULT * hidden) ** 0.5
    )
    if path == "A":
        x = (
            torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
            * 0.5
        )
        residual = x.unsqueeze(1).expand(num_tokens, HC_MULT, hidden).contiguous()
        fn = fn_full.view(MHC_MULT3, HC_MULT, hidden).sum(1)
        rms_numel = hidden
        pre_mix_in = None
    else:
        residual = (
            torch.randn(num_tokens, HC_MULT, hidden, device="cuda", dtype=torch.bfloat16)
            * 0.5
        )
        fn = fn_full
        x = residual.view(num_tokens, HC_MULT * hidden)
        rms_numel = HC_MULT * hidden
        pre_mix_in = torch.rand(num_tokens, HC_MULT, device="cuda", dtype=torch.float32)

    gemm_out_mul, gemm_out_sqrsum = make_gemm(x, fn, n_splits)
    norm_weight = None
    if norm:
        norm_weight = (
            torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.1 + 1.0
        ).bfloat16()

    return dict(
        x=x,
        fn=fn,
        gemm_out_mul=gemm_out_mul,
        gemm_out_sqrsum=gemm_out_sqrsum,
        hc_scale=hc_scale,
        hc_base=hc_base,
        residual=residual,
        norm_weight=norm_weight,
        pre_mix_in=pre_mix_in,
        save_pre_mix=True,
        rms_numel=rms_numel,
    )


def run_op(case, aux_out=None):
    num_tokens = case["residual"].shape[0]
    hidden = case["residual"].shape[-1]
    post_mix = torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(num_tokens, 16, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    pre_mix_out = (
        torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32)
        if case["save_pre_mix"] else None
    )
    mcoplib_ops.mhc_pre_big_fuse_out(
        case["gemm_out_mul"],
        case["gemm_out_sqrsum"],
        case["hc_scale"],
        case["hc_base"],
        case["residual"],
        post_mix,
        comb_mix,
        layer_input,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        case["gemm_out_mul"].shape[0],
        norm_weight=case["norm_weight"],
        pre_mix_in=case["pre_mix_in"],
        pre_mix_out=pre_mix_out,
        aux_out=aux_out,
        norm_eps=NORM_EPS,
        rms_numel=case["rms_numel"],
    )
    return post_mix, comb_mix, layer_input, pre_mix_out


def run_broadcast_op(case):
    """Run the first-layer [N,H] -> [N,4,H] broadcast contract."""
    residual = case["residual"][:, 0, :].contiguous()
    num_tokens, hidden = residual.shape
    post_mix = torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(num_tokens, 16, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    residual_out = torch.empty(
        num_tokens, 4, hidden, device="cuda", dtype=torch.bfloat16
    )
    mcoplib_ops.mhc_pre_big_fuse_out(
        case["gemm_out_mul"],
        case["gemm_out_sqrsum"],
        case["hc_scale"],
        case["hc_base"],
        residual,
        post_mix,
        comb_mix,
        layer_input,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        case["gemm_out_mul"].shape[0],
        norm_weight=case["norm_weight"],
        norm_eps=NORM_EPS,
        residual_out=residual_out,
    )
    return residual, residual_out, post_mix, comb_mix, layer_input


REFERENCE_NAMES = ("torch",) + (("vllm",) if VLLM_REFERENCE_AVAILABLE else ())


def _assert_matches(actual, expected, mixed_rtol=5e-3, layer_rtol=2e-2):
    post_mix, comb_mix, layer_input, pre_mix_out = actual
    exp_post, exp_comb, exp_layer, exp_pre, exp_aux = expected
    torch.testing.assert_close(post_mix, exp_post, rtol=mixed_rtol, atol=mixed_rtol)
    torch.testing.assert_close(comb_mix, exp_comb, rtol=mixed_rtol, atol=mixed_rtol)
    torch.testing.assert_close(
        pre_mix_out, exp_pre, rtol=mixed_rtol, atol=mixed_rtol
    )
    torch.testing.assert_close(
        layer_input.float(), exp_layer.float(), rtol=layer_rtol, atol=layer_rtol
    )
    return exp_aux


def _reference_kwargs(case, **overrides):
    kwargs = dict(
        gemm_out_mul=case["gemm_out_mul"],
        gemm_out_sqrsum=case["gemm_out_sqrsum"],
        hc_scale=case["hc_scale"],
        hc_base=case["hc_base"],
        residual=case["residual"],
        norm_weight=case["norm_weight"],
        pre_mix_in=case["pre_mix_in"],
        save_pre_mix=case["save_pre_mix"],
        rms_numel=case["rms_numel"],
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_eps=NORM_EPS,
        write_aux=False,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Shared delayed shape contract, including the logged 11-tensor signature.
# --------------------------------------------------------------------------
def _assert_delayed_shapes(case, actual):
    num_tokens = case["residual"].shape[0]
    n_splits = case["gemm_out_mul"].shape[0]
    gemm_hidden = HIDDEN if case["pre_mix_in"] is None else HC_MULT * HIDDEN
    expected_inputs = (
        ("x", (num_tokens, gemm_hidden), torch.bfloat16),
        ("fn", (MHC_MULT3, gemm_hidden), torch.float32),
        (
            "gemm_out_mul",
            (n_splits, num_tokens, MHC_MULT3),
            torch.float32,
        ),
        (
            "gemm_out_sqrsum",
            (n_splits, num_tokens),
            torch.float32,
        ),
        ("hc_scale", (3,), torch.float32),
        ("hc_base", (MHC_MULT3,), torch.float32),
        ("residual", (num_tokens, HC_MULT, HIDDEN), torch.bfloat16),
    )
    if case["norm_weight"] is not None:
        expected_inputs += (("norm_weight", (HIDDEN,), torch.bfloat16),)
    for name, shape, dtype in expected_inputs:
        tensor = case[name]
        assert tuple(tensor.shape) == shape, name
        assert tensor.dtype == dtype, name
        assert tensor.is_contiguous(), name
    assert case["rms_numel"] == gemm_hidden
    assert case["save_pre_mix"] is True
    if case["pre_mix_in"] is not None:
        assert tuple(case["pre_mix_in"].shape) == (num_tokens, HC_MULT)
        assert case["pre_mix_in"].dtype == torch.float32

    for tensor, width, dtype in zip(
        actual,
        (HC_MULT, HC_MULT * HC_MULT, HIDDEN, HC_MULT),
        (torch.float32, torch.float32, torch.bfloat16, torch.float32),
    ):
        assert tuple(tensor.shape) == (num_tokens, width)
        assert tensor.dtype == dtype


# --------------------------------------------------------------------------
# Accuracy: native vs vLLM original (and vs the torch cross-check).
# --------------------------------------------------------------------------
# Log cases share the accuracy runner with the extension cases. The extracted
# upstream norm kernel has an unresolved square-sum reduction discrepancy
# (sqrt(2) output scaling), so the log set uses the independent torch oracle.
# Optional upstream comparisons are retained in the extension set.
DSV41_FLASH_INT8_TP8DP2_CASES = tuple(
    (n, path, True, DSV41_FLASH_INT8_TP8DP2_SPLITS, "torch")
    for n in DSV41_FLASH_INT8_TP8DP2_TOKENS
    for path in ("A", "B")
)
DELAYED_ACCURACY_CASES = tuple(
    pytest.param(
        *case, id=f"dsv41_flash_int8_tp8dp2-Path{case[1]}-N{case[0]}"
    )
    for case in DSV41_FLASH_INT8_TP8DP2_CASES
) + tuple(
    pytest.param(
        n, path, norm, splits, reference,
        id=f"extension-Path{path}-N{n}-norm{int(norm)}-S{splits}-{reference}",
    )
    for n in NUM_TOKENS
    for path in ("A", "B")
    for norm in (False, True)
    for splits in (16 if n >= 512 else 4,)
    for reference in REFERENCE_NAMES
    if (n, path, norm, splits, reference) not in DSV41_FLASH_INT8_TP8DP2_CASES
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "num_tokens,path,norm,n_splits,reference", DELAYED_ACCURACY_CASES
)
def test_h5120_delayed_accuracy(num_tokens, path, norm, n_splits, reference):
    if reference == "vllm" and not VLLM_REFERENCE_AVAILABLE:
        pytest.skip(VLLM_SKIP_REASON)
    case = build_case(num_tokens, n_splits, path, norm, seed=20261010 + num_tokens)
    actual = run_op(case)
    _assert_delayed_shapes(case, actual)
    expected = run_reference(reference, **_reference_kwargs(case))
    if (num_tokens, path, norm, n_splits, reference) in DSV41_FLASH_INT8_TP8DP2_CASES:
        _assert_matches(actual, expected, mixed_rtol=1e-6, layer_rtol=1e-2)
    else:
        _assert_matches(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("reference", REFERENCE_NAMES)
@pytest.mark.parametrize("num_tokens", (1, 16, 99, 2048))
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
def test_h5120_split_sweep(num_tokens, n_splits, reference):
    if reference == "vllm" and not VLLM_REFERENCE_AVAILABLE:
        pytest.skip(VLLM_SKIP_REASON)
    case = build_case(num_tokens, n_splits, "B", True, seed=777 + n_splits)
    actual = run_op(case)
    expected = run_reference(reference, **_reference_kwargs(case))
    _assert_matches(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("reference", REFERENCE_NAMES)
@pytest.mark.parametrize("num_tokens", (16, 99))
def test_h5120_aux_output(num_tokens, reference):
    if reference == "vllm" and not VLLM_REFERENCE_AVAILABLE:
        pytest.skip(VLLM_SKIP_REASON)
    case = build_case(num_tokens, 4, "B", True, seed=4242 + num_tokens)
    aux_out = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    actual = run_op(case, aux_out=aux_out)
    expected = run_reference(
        reference, **_reference_kwargs(case, write_aux=True)
    )
    exp_aux = _assert_matches(actual, expected)
    torch.testing.assert_close(
        aux_out.float(), exp_aux.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and VLLM_REFERENCE_AVAILABLE),
    reason=VLLM_SKIP_REASON,
)
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize("norm", (False, True))
def test_torch_crosscheck_agrees_with_vllm(path, norm):
    """The independent torch reference must agree with the vLLM original."""
    num_tokens = 126
    case = build_case(num_tokens, 4, path, norm, seed=606 + num_tokens)
    kwargs = _reference_kwargs(case)
    vllm_out = vllm_reference(**kwargs)
    torch_out = torch_reference(**kwargs)
    torch.testing.assert_close(vllm_out[0], torch_out[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(vllm_out[1], torch_out[1], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        vllm_out[3], torch_out[3], rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        vllm_out[2].float(), torch_out[2].float(), rtol=2e-2, atol=2e-2
    )


# --------------------------------------------------------------------------
# Invariants and interface behaviour.
# --------------------------------------------------------------------------
def _refresh_edge_gemm(case):
    """Keep split GEMM inputs consistent after replacing the residual."""
    residual = case["residual"]
    case["x"] = (
        residual[:, 0, :].contiguous()
        if case["pre_mix_in"] is None
        else residual.reshape(residual.shape[0], -1)
    )
    splits = case["gemm_out_mul"].shape[0]
    case["gemm_out_mul"], case["gemm_out_sqrsum"] = make_gemm(
        case["x"], case["fn"], splits
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
@pytest.mark.parametrize("nonzeros", (0, 1, 256, 257, HIDDEN))
def test_h5120_norm_reduction_edges(path, n_splits, nonzeros):
    """Closed-form RMS checks detect missing lanes in the final reduction.

    256/257 straddle the previously observed upstream reduction failure;
    dense ones must normalize to BF16 1, without sqrt(2) scaling.
    """
    case = build_case(1, n_splits, path, True, seed=731)
    case["residual"].zero_()
    case["residual"][:, 0, :nonzeros] = 1
    case["norm_weight"].fill_(1)
    if path == "B":
        case["pre_mix_in"].zero_()
        case["pre_mix_in"][:, 0] = 1
    _refresh_edge_gemm(case)
    actual = run_op(case)
    expected = torch_reference(**_reference_kwargs(case))
    _assert_matches(actual, expected, mixed_rtol=1e-6)
    closed_form = torch.zeros_like(actual[2])
    closed_form[:, :nonzeros] = (nonzeros / HIDDEN + NORM_EPS) ** -0.5
    torch.testing.assert_close(actual[2], closed_form, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize(
    "position",
    (0, 255, 256, 1023, 1024, 3071, 3072, 3583, 3584, 4095, 4096, 5119),
)
def test_h5120_norm_spike_positions(path, position):
    """Exercise vector/tile boundaries, including the last two cached tiles."""
    case = build_case(1, 16, path, True, seed=732)
    case["residual"].zero_()
    case["residual"][:, 0, position] = 1
    case["norm_weight"].fill_(1)
    if path == "B":
        case["pre_mix_in"].zero_()
        case["pre_mix_in"][:, 0] = 1
    _refresh_edge_gemm(case)
    actual = run_op(case)
    expected = torch.zeros_like(actual[2])
    expected[:, position] = (1 / HIDDEN + NORM_EPS) ** -0.5
    torch.testing.assert_close(actual[2], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_h5120_stream0_cached_tail_boundaries_n2048():
    """Exercise the actual N2048 cache dispatch with a closed-form RMS oracle."""
    case = build_case(2048, 16, "A", True, seed=20261014)
    residual = case["residual"]
    residual.zero_()
    case["norm_weight"].fill_(1)
    boundaries = torch.tensor(
        (0, 511, 512, 1023, 1024, 1535, 1536, 2047, 2048,
         3071, 3072, 3583, 3584, 4095, 4096, 5119),
        device=residual.device, dtype=torch.long,
    )
    rows = torch.arange(residual.shape[0], device=residual.device)
    positions = boundaries[rows % boundaries.numel()]
    residual[rows, 0, positions] = 1
    _refresh_edge_gemm(case)
    actual = run_op(case)
    expected = torch.zeros_like(actual[2])
    expected[rows, positions] = (1 / HIDDEN + NORM_EPS) ** -0.5
    torch.testing.assert_close(actual[2], expected, rtol=0, atol=0)
    assert torch.equal(actual[2].view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
def test_h5120_carried_rounds_to_bf16_before_rms(n_splits):
    """1 + .0038 rounds to BF16 1 before RMS, including the last tile."""
    case = build_case(1, n_splits, "B", True, seed=733)
    case["residual"].zero_()
    case["residual"][:, :2, :] = 1
    case["norm_weight"].fill_(1)
    case["pre_mix_in"].zero_()
    case["pre_mix_in"][:, 0] = 1
    case["pre_mix_in"][:, 1] = 0.0038
    _refresh_edge_gemm(case)
    actual = run_op(case)
    torch.testing.assert_close(actual[2], torch.ones_like(actual[2]), rtol=0, atol=0)
    unrounded = (case["residual"].float() * case["pre_mix_in"].unsqueeze(-1)).sum(1)
    fp32_rms_variant = (
        unrounded.bfloat16().float()
        * torch.rsqrt(unrounded.square().mean(-1, keepdim=True) + NORM_EPS)
    ).bfloat16()
    assert not torch.equal(actual[2], fp32_rms_variant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
def test_h5120_carried_cancellation(n_splits):
    case = build_case(1, n_splits, "B", True, seed=734)
    case["residual"].zero_()
    case["residual"][:, 0, :] = 1
    case["residual"][:, 1, :] = -1
    case["pre_mix_in"].zero_()
    case["pre_mix_in"][:, :2] = 1
    _refresh_edge_gemm(case)
    actual = run_op(case)
    _assert_matches(actual, torch_reference(**_reference_kwargs(case)), mixed_rtol=1e-6)
    torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
@pytest.mark.parametrize("write_aux", (False, True))
def test_h5120_norm_without_saving_pre_mix(path, n_splits, write_aux):
    """Cover the other affected norm instance and its FP32-before-RMS rule.

    Without carried weights or pre_mix_out, collapse uses newly computed
    pre-mix weights; it is no longer the delayed one-hot path.
    """
    case = build_case(3, n_splits, path, True, seed=735)
    case["save_pre_mix"] = False
    aux = torch.empty_like(case["residual"][:, 0, :]) if write_aux else None
    actual = run_op(case, aux_out=aux)
    expected = torch_reference(**_reference_kwargs(case, write_aux=write_aux))
    for value, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-2, atol=1e-2)
    assert actual[3] is None
    if write_aux:
        torch.testing.assert_close(aux, expected[4], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n_splits", NUM_SPLITS)
def test_h5120_broadcast_with_norm(n_splits):
    """Match vLLM's first-layer broadcast big-fuse contract in one launch."""
    num_tokens = 3
    case = build_case(num_tokens, n_splits, "A", True, seed=811 + n_splits)
    case["save_pre_mix"] = False

    residual, residual_out, post, comb, layer_input = run_broadcast_op(case)
    expected_residual = residual.unsqueeze(1).expand(-1, HC_MULT, -1).contiguous()
    expected = torch_reference(
        **_reference_kwargs(
            case,
            residual=expected_residual,
            save_pre_mix=False,
            rms_numel=HIDDEN,
        )
    )

    assert torch.equal(residual_out, expected_residual)
    torch.testing.assert_close(post, expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(comb, expected[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        layer_input.float(), expected[2].float(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize("num_tokens", (1, 4, 99, 2048))
def test_h5120_control_header_repeated_updates(num_tokens, path):
    """Changing control data exposes stale reads at the warp handoff."""
    case = build_case(num_tokens, 16, path, True, seed=20261012 + num_tokens)
    for _ in range(8):
        case["gemm_out_mul"].normal_(mean=0.0, std=0.25)
        case["gemm_out_sqrsum"].uniform_(512.0, 4096.0)
        actual = run_op(case)
        expected = torch_reference(**_reference_kwargs(case))
        _assert_matches(actual, expected, mixed_rtol=1e-6, layer_rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("path", ("A", "B"))
@pytest.mark.parametrize("num_tokens", (1, 99, 2048))
@pytest.mark.parametrize("gemm_last_dim", (25, 32))
def test_h5120_split16_padded_gemm_stride(num_tokens, path, gemm_last_dim):
    """Fixed split count must still honor the actual [S,N,L] row stride."""
    case = build_case(num_tokens, 16, path, True, seed=20261012 + num_tokens)
    expected = torch_reference(**_reference_kwargs(case))
    packed_actual = run_op(case)
    _assert_matches(packed_actual, expected, mixed_rtol=1e-6, layer_rtol=1e-2)

    # NaN padding exposes reads from the wrong token/split or extra mix columns.
    padded = torch.full(
        (16, num_tokens, gemm_last_dim), float("nan"),
        device="cuda", dtype=torch.float32,
    )
    padded[:, :, :MHC_MULT3].copy_(case["gemm_out_mul"])
    padded_case = dict(case, gemm_out_mul=padded)
    padded_actual = run_op(padded_case)
    _assert_matches(padded_actual, expected, mixed_rtol=1e-6, layer_rtol=1e-2)
    for packed_output, padded_output in zip(packed_actual, padded_actual):
        assert torch.equal(packed_output, padded_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_tokens", (1, 99, 2048))
@pytest.mark.parametrize("gemm_last_dim", (24, 32))
def test_h5120_input_mode_switch_with_reused_outputs(num_tokens, gemm_last_dim):
    """Switch carried/stream-zero modes; unused streams must not poison one-hot."""
    case = build_case(num_tokens, 16, "B", True, seed=20261013 + num_tokens)
    finite_residual = case["residual"].clone()
    weights = torch.tensor(
        [0.25, -0.75, 0.0, 1.125], device="cuda", dtype=torch.float32,
    ).expand(num_tokens, 4).contiguous()
    if gemm_last_dim > MHC_MULT3:
        padded = torch.full(
            (16, num_tokens, gemm_last_dim), float("nan"),
            device="cuda", dtype=torch.float32,
        )
        padded[:, :, :MHC_MULT3].copy_(case["gemm_out_mul"])
        case["gemm_out_mul"] = padded
    outputs = (
        torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32),
        torch.empty(num_tokens, 16, device="cuda", dtype=torch.float32),
        torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16),
        torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32),
    )
    for step in range(6):
        carried = step % 2 == 1
        case["residual"].copy_(finite_residual)
        if not carried:
            case["residual"][:, 1:, :].fill_(float("nan"))
        case["pre_mix_in"] = weights if carried else None
        case["rms_numel"] = HC_MULT * HIDDEN if carried else HIDDEN
        for output in outputs:
            output.fill_(float("nan"))
        mcoplib_ops.mhc_pre_big_fuse_out(
            case["gemm_out_mul"], case["gemm_out_sqrsum"],
            case["hc_scale"], case["hc_base"], case["residual"],
            *outputs[:3], RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE, SINKHORN_REPEAT, 16,
            norm_weight=case["norm_weight"], pre_mix_in=case["pre_mix_in"],
            pre_mix_out=outputs[3], norm_eps=NORM_EPS,
            rms_numel=case["rms_numel"],
        )
        # Dense Torch 0*NaN would poison unused streams. The finite reference
        # has the identical active stream and implements the one-hot contract.
        expected = torch_reference(
            **_reference_kwargs(
                case, residual=finite_residual,
                gemm_out_mul=case["gemm_out_mul"][:, :, :MHC_MULT3],
            ),
        )
        _assert_matches(outputs, expected, mixed_rtol=1e-6, layer_rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_path_a_and_b_mixes_agree():
    """Path A/B must agree on post/comb/pre_mix_out when the residual is the
    broadcast of the projection input and fn_A is the channel sum of fn_B."""
    num_tokens = 123
    n_splits = 4
    hidden = HIDDEN
    torch.manual_seed(31337)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32) * 0.2 + 1.0
    hc_base = torch.randn(MHC_MULT3, device="cuda", dtype=torch.float32) * 0.2
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    residual = x.unsqueeze(1).expand(num_tokens, HC_MULT, hidden).contiguous()
    fn_full = (
        torch.randn(MHC_MULT3, HC_MULT * hidden, device="cuda", dtype=torch.float32)
        / (HC_MULT * hidden) ** 0.5
    )
    fn_broadcast = fn_full.view(MHC_MULT3, HC_MULT, hidden).sum(1)
    norm_weight = (
        torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.1 + 1.0
    ).bfloat16()

    mul_a, sq_a = make_gemm(x, fn_broadcast, n_splits)
    mul_b, sq_b = make_gemm(
        residual.view(num_tokens, HC_MULT * hidden), fn_full, n_splits
    )

    case_a = dict(
        gemm_out_mul=mul_a,
        gemm_out_sqrsum=sq_a,
        hc_scale=hc_scale,
        hc_base=hc_base,
        residual=residual,
        norm_weight=norm_weight,
        pre_mix_in=None,
        save_pre_mix=True,
        rms_numel=hidden,
    )
    carried = torch.rand(num_tokens, HC_MULT, device="cuda", dtype=torch.float32)
    case_b = dict(case_a)
    case_b.update(
        gemm_out_mul=mul_b,
        gemm_out_sqrsum=sq_b,
        pre_mix_in=carried,
        rms_numel=HC_MULT * hidden,
    )

    post_a, comb_a, layer_a, pre_a = run_op(case_a)
    post_b, comb_b, layer_b, pre_b = run_op(case_b)

    torch.testing.assert_close(post_a, post_b, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(comb_a, comb_b, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(pre_a, pre_b, rtol=5e-3, atol=5e-3)
    assert not torch.allclose(layer_a.float(), layer_b.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_pre_mix_in_only_affects_layer_input():
    num_tokens = 61
    case = build_case(num_tokens, 4, "B", True, seed=5150)
    other = torch.rand(num_tokens, HC_MULT, device="cuda", dtype=torch.float32)
    post_a, comb_a, layer_a, _ = run_op(case)
    case_b = dict(case)
    case_b["pre_mix_in"] = other
    post_b, comb_b, layer_b, _ = run_op(case_b)
    torch.testing.assert_close(post_a, post_b, rtol=0, atol=0)
    torch.testing.assert_close(comb_a, comb_b, rtol=0, atol=0)
    assert not torch.allclose(layer_a.float(), layer_b.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_h5120_zero_tokens_is_noop():
    residual = torch.empty(0, HC_MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    gemm_out_mul = torch.empty(4, 0, MHC_MULT3, device="cuda", dtype=torch.float32)
    gemm_out_sqrsum = torch.empty(4, 0, device="cuda", dtype=torch.float32)
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(MHC_MULT3, device="cuda", dtype=torch.float32)
    post_mix = torch.empty(0, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(0, HIDDEN, device="cuda", dtype=torch.bfloat16)
    mcoplib_ops.mhc_pre_big_fuse_out(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        4,
        rms_numel=HIDDEN,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_h5120_rejects_bad_shapes():
    case = build_case(8, 4, "B", True, seed=99)
    post_mix = torch.empty(8, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(8, 16, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(8, HIDDEN, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError):
        # n_splits outside {1, 4, 16}.
        mcoplib_ops.mhc_pre_big_fuse_out(
            case["gemm_out_mul"][:2],
            case["gemm_out_sqrsum"][:2],
            case["hc_scale"],
            case["hc_base"],
            case["residual"],
            post_mix,
            comb_mix,
            layer_input,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            2,
            rms_numel=HC_MULT * HIDDEN,
        )

    with pytest.raises(RuntimeError):
        # rms_numel must be 5120 or 20480.
        mcoplib_ops.mhc_pre_big_fuse_out(
            case["gemm_out_mul"],
            case["gemm_out_sqrsum"],
            case["hc_scale"],
            case["hc_base"],
            case["residual"],
            post_mix,
            comb_mix,
            layer_input,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            4,
            rms_numel=1234,
        )

    with pytest.raises(RuntimeError):
        # layer_input aliasing the residual input is rejected.
        alias = case["residual"].view(8, -1)[:, :HIDDEN]
        mcoplib_ops.mhc_pre_big_fuse_out(
            case["gemm_out_mul"],
            case["gemm_out_sqrsum"],
            case["hc_scale"],
            case["hc_base"],
            case["residual"],
            post_mix,
            comb_mix,
            alias,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            4,
            rms_numel=HC_MULT * HIDDEN,
        )


def _call_h5120_views(case, post_mix, comb_mix, layer_input, pre_mix_out):
    """Exercise the public entry with caller-owned contiguous storage views."""
    mcoplib_ops.mhc_pre_big_fuse_out(
        case["gemm_out_mul"], case["gemm_out_sqrsum"], case["hc_scale"],
        case["hc_base"], case["residual"], post_mix, comb_mix, layer_input,
        RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT_VALUE,
        SINKHORN_REPEAT, case["gemm_out_mul"].shape[0],
        norm_weight=case["norm_weight"], pre_mix_in=case["pre_mix_in"],
        pre_mix_out=pre_mix_out, norm_eps=NORM_EPS,
        rms_numel=case["rms_numel"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("overlap", ("post_gemm", "layer_residual", "post_comb"))
def test_h5120_contiguous_overlap_rejected(overlap):
    case = build_case(1, 16, "B", True, seed=20261013)
    post = torch.empty(1, 4, device="cuda")
    comb = torch.empty(1, 16, device="cuda")
    layer = torch.empty(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pre = torch.empty(1, 4, device="cuda")
    message = "outputs must not alias inputs"
    if overlap == "post_gemm":
        post = case["gemm_out_mul"].view(-1)[8:12].view(1, 4)
    elif overlap == "layer_residual":
        layer = case["residual"].view(-1)[8:8 + HIDDEN].view(1, HIDDEN)
    else:
        storage = torch.empty(32, device="cuda")
        post, comb = storage[8:12].view(1, 4), storage[4:20].view(1, 16)
        message = "outputs must not alias each other"
    assert all(t.is_contiguous() for t in (post, comb, layer))
    assert (layer if overlap == "layer_residual" else post).storage_offset() > 0
    with pytest.raises(RuntimeError, match=message):
        _call_h5120_views(case, post, comb, layer, pre)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("gap", (0, 8))
def test_h5120_same_storage_disjoint_views(gap):
    case = build_case(1, 16, "B", True, seed=20261013)
    expected = torch_reference(**_reference_kwargs(case))
    fp32 = torch.full((8 + 24 + gap + 4 + gap + 16 + 8,), float("nan"), device="cuda")
    base = case["hc_base"]
    case["hc_base"] = fp32[8:32]
    case["hc_base"].copy_(base)
    post_begin = 32 + gap
    comb_begin = post_begin + 4 + gap
    post = fp32[post_begin:post_begin + 4].view(1, 4)
    comb = fp32[comb_begin:comb_begin + 16].view(1, 16)
    bf16 = torch.full((8 + 6 * HIDDEN + 2 * gap + 8,), float("nan"),
                      device="cuda", dtype=torch.bfloat16)
    residual, norm = case["residual"], case["norm_weight"]
    case["residual"] = bf16[8:8 + 4 * HIDDEN].view(1, 4, HIDDEN)
    case["residual"].copy_(residual)
    norm_begin = 8 + 4 * HIDDEN + gap
    case["norm_weight"] = bf16[norm_begin:norm_begin + HIDDEN]
    case["norm_weight"].copy_(norm)
    layer_begin = norm_begin + HIDDEN + gap
    layer = bf16[layer_begin:layer_begin + HIDDEN].view(1, HIDDEN)
    pre = torch.empty(1, 4, device="cuda")
    assert all(t.is_contiguous() and t.storage_offset() > 0
               for t in (case["hc_base"], post, comb, case["residual"], case["norm_weight"], layer))
    _call_h5120_views(case, post, comb, layer, pre)
    _assert_matches((post, comb, layer, pre), expected, mixed_rtol=1e-6, layer_rtol=1e-2)
    assert torch.isnan(fp32[:8]).all() and torch.isnan(fp32[-8:]).all()
    assert torch.isnan(bf16[:8]).all() and torch.isnan(bf16[-8:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("overlap", (
    "pre_carried", "pre_norm", "pre_residual", "pre_layer",
    "post_norm", "comb_norm", "post_carried", "comb_carried",
))
def test_h5120_optional_alias_fallback_views(overlap):
    """Allowed optional overlaps must keep the existing sequential fallback.

    The frozen-extension validator restores aliased inputs before its baseline
    call and checks every output bitwise. These overlaps can overwrite inputs;
    they are not assigned new mathematical semantics by this regression test.
    """
    case = build_case(1, 16, "B", True, seed=20261013)
    post = torch.empty(1, 4, device="cuda")
    comb = torch.empty(1, 16, device="cuda")
    layer = torch.empty(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pre = torch.empty(1, 4, device="cuda")
    if overlap == "pre_carried":
        pre = case["pre_mix_in"]
    elif overlap == "pre_norm":
        pre = case["norm_weight"].view(torch.float32)[8:12].view(1, 4)
    elif overlap == "pre_residual":
        pre = case["residual"].view(-1).view(torch.float32)[8:12].view(1, 4)
    elif overlap == "pre_layer":
        pre = layer.view(torch.float32)[:, 8:12]
    elif overlap == "post_norm":
        post = case["norm_weight"].view(torch.float32)[8:12].view(1, 4)
    elif overlap == "comb_norm":
        comb = case["norm_weight"].view(torch.float32)[8:24].view(1, 16)
    else:
        storage = torch.empty(32, device="cuda")
        carried = case["pre_mix_in"]
        case["pre_mix_in"] = storage[8:12].view(1, 4)
        case["pre_mix_in"].copy_(carried)
        if overlap == "post_carried":
            post = storage[10:14].view(1, 4)
        else:
            comb = storage[4:20].view(1, 16)
    assert all(t.is_contiguous() for t in (post, comb, layer, pre))
    _call_h5120_views(case, post, comb, layer, pre)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_h5120_layer_norm_exact_alias_keeps_forward_order():
    """The allowed layer/norm alias must retain the original block order."""
    case = build_case(1, 16, "A", True, seed=20261014)
    expected = torch_reference(**_reference_kwargs(case))
    post = torch.empty(1, 4, device="cuda")
    comb = torch.empty(1, 16, device="cuda")
    pre = torch.empty(1, 4, device="cuda")
    layer = case["norm_weight"].view(1, HIDDEN)
    _call_h5120_views(case, post, comb, layer, pre)
    _assert_matches(
        (post, comb, layer, pre), expected, mixed_rtol=1e-6, layer_rtol=1e-2
    )


# --------------------------------------------------------------------------
# Legacy path regression guard (hidden 4096/7168 unchanged).
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("hidden", (4096, 7168))
def test_legacy_path_still_works(hidden):
    num_tokens = 16
    n_splits = 16
    torch.manual_seed(1234 + hidden)
    residual = torch.randn(
        num_tokens, HC_MULT, hidden, device="cuda", dtype=torch.bfloat16
    )
    fn = (
        torch.randn(MHC_MULT3, HC_MULT * hidden, device="cuda", dtype=torch.float32)
        / (HC_MULT * hidden) ** 0.5
    )
    gemm_out_mul, gemm_out_sqrsum = make_gemm(
        residual.view(num_tokens, HC_MULT * hidden), fn, n_splits
    )
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32) * 0.2 + 1.0
    hc_base = torch.randn(MHC_MULT3, device="cuda", dtype=torch.float32) * 0.2
    post_mix = torch.empty(num_tokens, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(num_tokens, 16, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    mcoplib_ops.mhc_pre_big_fuse_out(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        n_splits,
    )
    expected = torch_reference(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        None,
        None,
        False,
        HC_MULT * hidden,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        NORM_EPS,
        False,
    )
    torch.testing.assert_close(post_mix, expected[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(comb_mix, expected[1], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        layer_input.float(), expected[2].float(), rtol=2e-2, atol=2e-2
    )
