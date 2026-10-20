import pytest
import torch

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "dsv3_fused_a_gemm"

# C++ dispatcher 中无条件支持的生产 shape：(hd_in, hd_out)
SUPPORTED_SHAPES = [
    (7168, 1536),
    (7168, 2112),
    (1536, 2304),
    (1536, 4608),
    (7168, 3584),
    (768, 7168),
    (1536, 1152),
    (7168, 768),
    (7168, 3216),
    (7168, 4224),
    (6144, 2624),
    (2048, 2048),
    (128, 1536),
    (128, 3072),
    (128, 768),
    (384, 7168),
]

TOKEN_COUNTS = [1, 2, 4, 8, 15, 16]


def _require_dsv3():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not registered")


def _make_input(shape, seed, mode="randn"):
    # 在 CPU 上使用独立 generator，确保测试可重复。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if mode == "zeros":
        tensor = torch.zeros(shape, dtype=torch.bfloat16)
    elif mode == "ones":
        tensor = torch.ones(shape, dtype=torch.bfloat16)
    elif mode == "randint":
        tensor = torch.randint(
            -3, 4, shape, generator=generator, dtype=torch.int32
        ).to(torch.bfloat16)
    else:
        tensor = torch.randn(
            shape, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)

    return tensor.to(DEVICE)


def _make_case(hd_in, hd_out, num_tokens, seed, mode="randn"):
    x = _make_input((num_tokens, hd_in), seed, mode)

    # 原始 weight 为连续 [hd_out, hd_in]。
    # 转置后得到 [hd_in, hd_out]，stride 为 (1, hd_in)。
    weight = _make_input((hd_out, hd_in), seed + 100003, mode)
    weight_t = weight.t()

    output = torch.empty(
        (num_tokens, hd_out),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    return x, weight_t, output


def _call(output, x, weight_t, enable_pdl):
    torch.ops._C.dsv3_fused_a_gemm(
        output, x, weight_t, enable_pdl=enable_pdl
    )


def _assert_output_metadata(output, num_tokens, hd_out):
    assert output.shape == (num_tokens, hd_out)
    assert output.dtype == torch.bfloat16
    assert output.device.type == "cuda"
    assert output.stride() == (hd_out, 1)


# ----------------------------------------------------------------------
# 1. 2048 组布局与接口调用测试
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(2048))
def test_dsv3_fused_a_gemm_layout(seed):
    _require_dsv3()

    num_tokens = 1
    hd_in = 128
    hd_out = 1536

    x, weight_t, output = _make_case(
        hd_in, hd_out, num_tokens, seed
    )

    assert x.shape == (num_tokens, hd_in)
    assert x.stride() == (hd_in, 1)

    assert weight_t.shape == (hd_in, hd_out)
    assert weight_t.stride() == (1, hd_in)

    assert output.shape == (num_tokens, hd_out)
    assert output.stride() == (hd_out, 1)

    _call(output, x, weight_t, enable_pdl=bool(seed % 2))
    torch.cuda.synchronize()

    _assert_output_metadata(output, num_tokens, hd_out)


# ----------------------------------------------------------------------
# 2. 所有生产 shape、token 数量及 PDL 开关
# ----------------------------------------------------------------------

@pytest.mark.parametrize("hd_in,hd_out", SUPPORTED_SHAPES)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_dsv3_fused_a_gemm_supported_shapes(
    hd_in, hd_out, num_tokens, enable_pdl
):
    _require_dsv3()

    x, weight_t, output = _make_case(
        hd_in, hd_out, num_tokens, seed=hd_in + hd_out
    )

    assert x.shape == (num_tokens, hd_in)
    assert x.stride(1) == 1
    assert num_tokens == 1 or x.stride(0) % 8 == 0

    assert weight_t.shape == (hd_in, hd_out)
    assert weight_t.stride() == (1, hd_in)

    _assert_output_metadata(output, num_tokens, hd_out)

    _call(output, x, weight_t, enable_pdl)
    torch.cuda.synchronize()

    _assert_output_metadata(output, num_tokens, hd_out)


# ----------------------------------------------------------------------
# 3. 小 shape 的输入模式覆盖
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["zeros", "ones", "randn", "randint"])
@pytest.mark.parametrize("num_tokens", [1, 2, 8, 16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_dsv3_fused_a_gemm_input_modes(
    mode, num_tokens, enable_pdl, seed
):
    _require_dsv3()

    x, weight_t, output = _make_case(
        128, 1536, num_tokens, seed, mode
    )

    _call(output, x, weight_t, enable_pdl)
    torch.cuda.synchronize()

    _assert_output_metadata(output, num_tokens, 1536)


# ----------------------------------------------------------------------
# 4. 非法 token 数量
# ----------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", [0, 17, 32])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_dsv3_fused_a_gemm_rejects_invalid_token_count(
    num_tokens, enable_pdl
):
    _require_dsv3()

    x = torch.empty(
        (num_tokens, 128), dtype=torch.bfloat16, device=DEVICE
    )
    weight_t = torch.empty(
        (128, 1536), dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        (num_tokens, 1536), dtype=torch.bfloat16, device=DEVICE
    )

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, enable_pdl)


# ----------------------------------------------------------------------
# 5. 非法输出 shape
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "output_shape",
    [
        (1, 1535),
        (1, 1537),
        (2, 1536),
        (0, 1536),
    ],
)
def test_dsv3_fused_a_gemm_rejects_output_shape(output_shape):
    _require_dsv3()

    x = torch.empty((1, 128), dtype=torch.bfloat16, device=DEVICE)
    weight_t = torch.empty(
        (128, 1536), dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        output_shape, dtype=torch.bfloat16, device=DEVICE
    )

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)


# ----------------------------------------------------------------------
# 6. 非法输入维度
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 127),
        (1, 129),
        (2, 127),
    ],
)
def test_dsv3_fused_a_gemm_rejects_input_shape(input_shape):
    _require_dsv3()

    x = torch.empty(input_shape, dtype=torch.bfloat16, device=DEVICE)
    weight_t = torch.empty(
        (128, 1536), dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        (input_shape[0], 1536),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)


# ----------------------------------------------------------------------
# 7. 非法权重维度
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "weight_shape",
    [
        (127, 1536),
        (129, 1536),
        (128, 1535),
    ],
)
def test_dsv3_fused_a_gemm_rejects_weight_shape(weight_shape):
    _require_dsv3()

    x = torch.empty((1, 128), dtype=torch.bfloat16, device=DEVICE)
    weight_t = torch.empty(
        weight_shape, dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        (1, 1536), dtype=torch.bfloat16, device=DEVICE
    )

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)


# ----------------------------------------------------------------------
# 8. 非法 dtype
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["input", "weight", "output"])
def test_dsv3_fused_a_gemm_rejects_wrong_dtype(which):
    _require_dsv3()

    x = torch.empty((1, 128), dtype=torch.bfloat16, device=DEVICE)
    weight_t = torch.empty(
        (128, 1536), dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        (1, 1536), dtype=torch.bfloat16, device=DEVICE
    )

    if which == "input":
        x = x.to(torch.float16)
    elif which == "weight":
        weight_t = weight_t.to(torch.float16)
    else:
        output = output.to(torch.float16)

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)


# ----------------------------------------------------------------------
# 9. 非二维 Tensor
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["input", "weight", "output"])
def test_dsv3_fused_a_gemm_rejects_non_2d_tensor(which):
    _require_dsv3()

    x = torch.empty((1, 128), dtype=torch.bfloat16, device=DEVICE)
    weight_t = torch.empty(
        (128, 1536), dtype=torch.bfloat16, device=DEVICE
    )
    output = torch.empty(
        (1, 1536), dtype=torch.bfloat16, device=DEVICE
    )

    if which == "input":
        x = x.reshape(-1)
    elif which == "weight":
        weight_t = weight_t.reshape(-1)
    else:
        output = output.reshape(-1)

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)


# ----------------------------------------------------------------------
# 10. 非法 stride / 内存布局
# ----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["input", "weight", "output"])
def test_dsv3_fused_a_gemm_rejects_invalid_layout(which):
    _require_dsv3()

    num_tokens = 2
    hd_in = 128
    hd_out = 1536

    x = torch.empty(
        (num_tokens, hd_in), dtype=torch.bfloat16, device=DEVICE
    )
    weight = torch.empty(
        (hd_out, hd_in), dtype=torch.bfloat16, device=DEVICE
    )
    weight_t = weight.t()
    output = torch.empty(
        (num_tokens, hd_out), dtype=torch.bfloat16, device=DEVICE
    )

    if which == "input":
        # 内层 stride 为 2，不满足 stride(1) == 1。
        x = torch.empty(
            (num_tokens, hd_in * 2),
            dtype=torch.bfloat16,
            device=DEVICE,
        )[:, ::2]

    elif which == "weight":
        # shape 正确，但不是要求的转置列主序布局。
        weight_t = torch.empty(
            (hd_in, hd_out),
            dtype=torch.bfloat16,
            device=DEVICE,
        )

    else:
        # shape 正确，但 output 的行 stride 带 padding。
        output = torch.empty(
            (num_tokens, hd_out + 8),
            dtype=torch.bfloat16,
            device=DEVICE,
        )[:, :hd_out]

    with pytest.raises(RuntimeError):
        _call(output, x, weight_t, True)