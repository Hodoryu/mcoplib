import pytest
import torch
import mcoplib._C


DEVICE = "cuda"


def _require_hisparse_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    try:
        return torch.ops._C.hisparse_gather_plan
    except AttributeError:
        pytest.skip("hisparse_gather_plan is not registered")


def _make_plan(rows, top_k, device=DEVICE):
    global_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
    hot_indices = torch.zeros_like(global_indices)
    miss_mask = torch.ones_like(global_indices)
    return global_indices, hot_indices, miss_mask


def _run_gather(host_cache, hot_cache, global_indices, hot_indices, miss_mask, request_state_indices=None, attention_indices=None, attention_block_stride=0):
    op = _require_hisparse_ops()
    op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, request_state_indices, attention_indices, attention_block_stride)
    torch.accelerator.synchronize()


@pytest.mark.parametrize("row_bytes", [16, 32, 64, 128])
@pytest.mark.parametrize("num_rows", [1, 2, 4])
@torch.no_grad()
def test_hisparse_gather_plan_zeroes_miss_rows(row_bytes, num_rows):
    op = _require_hisparse_ops()
    host_cache = torch.arange(num_rows * row_bytes, dtype=torch.uint8).reshape(num_rows, row_bytes).pin_memory()
    hot_cache = torch.full((num_rows, row_bytes), 7, dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(num_rows, 1)

    global_indices[:, 0] = num_rows
    hot_indices[:, 0] = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    miss_mask[:, 0] = 1

    op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)
    torch.accelerator.synchronize()

    torch.testing.assert_close(hot_cache, torch.zeros_like(hot_cache))


@pytest.mark.parametrize("row_bytes", [16, 32, 64])
@torch.no_grad()
def test_hisparse_gather_plan_copies_valid_host_rows(row_bytes):
    op = _require_hisparse_ops()
    num_rows = 4
    host_cache = torch.arange(num_rows * row_bytes, dtype=torch.uint8).reshape(num_rows, row_bytes).pin_memory()
    hot_cache = torch.full((num_rows, row_bytes), 7, dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(num_rows, 1)

    global_indices[:, 0] = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    hot_indices[:, 0] = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    miss_mask[:, 0] = 1

    op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)
    torch.accelerator.synchronize()

    expected = host_cache.to(DEVICE)
    torch.testing.assert_close(hot_cache, expected)


@torch.no_grad()
def test_hisparse_gather_plan_zeroes_unaligned_destination():
    op = _require_hisparse_ops()
    row_bytes = 16
    host_cache = torch.ones((1, row_bytes), dtype=torch.uint8).pin_memory()
    storage = torch.full((2 * row_bytes + 1,), 7, dtype=torch.uint8, device=DEVICE)
    hot_cache = torch.as_strided(storage, size=(2, 1, row_bytes), stride=(row_bytes + 1, row_bytes, 1))
    global_indices = torch.tensor([[1]], dtype=torch.int32, device=DEVICE)
    hot_indices = torch.tensor([[1]], dtype=torch.int32, device=DEVICE)
    miss_mask = torch.ones_like(global_indices)

    op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)
    torch.accelerator.synchronize()

    torch.testing.assert_close(hot_cache[1, 0], torch.zeros_like(hot_cache[1, 0]))
    assert storage[row_bytes].item() == 7


@torch.no_grad()
def test_hisparse_gather_plan_rejects_short_attention_block_stride():
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 16), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 2, 16), dtype=torch.uint8, device=DEVICE)
    global_indices = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    hot_indices = torch.zeros_like(global_indices)
    miss_mask = torch.ones_like(global_indices)
    attention_indices = torch.empty_like(global_indices)

    with pytest.raises(RuntimeError, match="stride must cover one hot block"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, attention_indices, 1)


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.int16])
@torch.no_grad()
def test_hisparse_gather_plan_rejects_invalid_plan_dtype(bad_dtype):
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 16), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 16), dtype=torch.uint8, device=DEVICE)
    global_indices = torch.zeros((1, 1), dtype=bad_dtype, device=DEVICE)
    hot_indices = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    miss_mask = torch.ones((1, 1), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="plan tensors must be int32"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)


@torch.no_grad()
def test_hisparse_gather_plan_rejects_non_pinned_host_cache():
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 16), dtype=torch.uint8)
    hot_cache = torch.empty((1, 16), dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="host_cache must be pinned CPU memory"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)


@torch.no_grad()
def test_hisparse_gather_plan_rejects_unaligned_row_bytes():
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 17), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 17), dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="KV rows must be 16-byte aligned"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)


@torch.no_grad()
def test_hisparse_gather_plan_rejects_mismatched_plan_shapes():
    op = _require_hisparse_ops()
    host_cache = torch.ones((2, 16), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((2, 16), dtype=torch.uint8, device=DEVICE)
    global_indices = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    hot_indices = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    miss_mask = torch.ones((2, 1), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="hot_indices/miss_mask must match contiguous 2D global_indices"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)


@torch.no_grad()
def test_hisparse_gather_plan_rejects_invalid_request_state_indices():
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 16), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 16), dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(1, 1)
    request_state_indices = torch.zeros((1,), dtype=torch.int64, device=DEVICE)

    with pytest.raises(RuntimeError, match="request_state_indices must be int32 on CUDA"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, request_state_indices, None, 0)


@torch.no_grad()
def test_hisparse_gather_plan_rejects_invalid_attention_indices():
    op = _require_hisparse_ops()
    host_cache = torch.ones((1, 16), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 16), dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(1, 1)
    attention_indices = torch.zeros((1, 1), dtype=torch.int64, device=DEVICE)

    with pytest.raises(RuntimeError, match="attention_indices must be a contiguous int32 CUDA tensor"):
        op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, attention_indices, 1)


@pytest.mark.parametrize("rows,top_k", [(0, 1), (1, 0), (0, 0)])
@torch.no_grad()
def test_hisparse_gather_plan_empty_input(rows, top_k):
    op = _require_hisparse_ops()
    row_bytes = 16
    host_cache = torch.empty((1, row_bytes), dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((max(rows, 1), row_bytes), dtype=torch.uint8, device=DEVICE)
    global_indices, hot_indices, miss_mask = _make_plan(rows, top_k)

    op(host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0)
    torch.accelerator.synchronize()