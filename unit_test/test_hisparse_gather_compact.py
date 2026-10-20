import pytest
import torch
import mcoplib._C


DEVICE = "cuda"


def _require_hisparse_compact_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    try:
        return torch.ops._C.hisparse_gather_compact
    except AttributeError:
        pytest.skip("hisparse_gather_compact is not registered in torch.ops._C")


def _make_host_cache(num_rows, row_bytes):
    return torch.arange(num_rows * row_bytes, dtype=torch.uint8).reshape(num_rows, row_bytes).pin_memory()


def _make_hot_cache(num_blocks, block_size, row_bytes, fill_value=7, device=DEVICE):
    return torch.full((num_blocks, block_size, row_bytes), fill_value, dtype=torch.uint8, device=device)


def _make_plan(num_rows, top_k, miss_counts=None, device=DEVICE):
    miss_global_indices = torch.zeros((num_rows, top_k), dtype=torch.int32, device=device)
    miss_hot_indices = torch.zeros((num_rows, top_k), dtype=torch.int32, device=device)
    if miss_counts is None:
        miss_counts = torch.full((num_rows,), top_k, dtype=torch.int32, device=device)
    else:
        miss_counts = torch.as_tensor(miss_counts, dtype=torch.int32, device=device).contiguous()
    return miss_global_indices, miss_hot_indices, miss_counts


def _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts):
    op = _require_hisparse_compact_op()
    op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    torch.accelerator.synchronize()


def _fill_valid_plan(miss_global_indices, miss_hot_indices, num_host_rows, num_hot_rows):
    num_rows, top_k = miss_global_indices.shape
    assert num_host_rows > 0
    assert num_hot_rows > 0
    assert num_rows * top_k <= num_host_rows * num_hot_rows or top_k >= 0

    for row in range(num_rows):
        for col in range(top_k):
            miss_global_indices[row, col] = (row * max(top_k, 1) + col) % num_host_rows
            miss_hot_indices[row, col] = (row * max(top_k, 1) + col) % num_hot_rows


def _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts):
    expected = hot_cache.clone()
    expected_flat = expected.reshape(-1, expected.shape[-1])
    global_cpu = miss_global_indices.cpu()
    hot_cpu = miss_hot_indices.cpu()
    counts_cpu = miss_counts.cpu()

    for row in range(miss_global_indices.shape[0]):
        count = int(counts_cpu[row].item())
        for col in range(count):
            src = int(global_cpu[row, col].item())
            dst = int(hot_cpu[row, col].item())
            expected_flat[dst].copy_(host_cache[src].to(expected.device))

    return expected


@pytest.mark.parametrize("row_bytes", [16, 32, 64, 128])
@pytest.mark.parametrize("num_rows", [1, 2, 4])
@torch.no_grad()
def test_hisparse_gather_compact_copies_all_misses(row_bytes, num_rows):
    _require_hisparse_compact_op()
    top_k = 2
    num_host_rows = 16
    block_size = 4
    num_blocks = 4
    num_hot_rows = block_size * num_blocks

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    hot_cache = _make_hot_cache(num_blocks, block_size, row_bytes)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k)

    for row in range(num_rows):
        for col in range(top_k):
            miss_global_indices[row, col] = (row * top_k + col + 1) % num_host_rows
            miss_hot_indices[row, col] = (row * top_k + col) % num_hot_rows
    miss_counts.fill_(top_k)

    expected = _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, expected)


@pytest.mark.parametrize("row_bytes", [16, 32, 64])
@torch.no_grad()
def test_hisparse_gather_compact_zero_miss_counts_preserves_cache(row_bytes):
    _require_hisparse_compact_op()
    num_rows = 4
    top_k = 3
    num_host_rows = 8
    block_size = 4
    num_blocks = 2

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    hot_cache = _make_hot_cache(num_blocks, block_size, row_bytes, fill_value=123)
    before = hot_cache.clone()
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k, [0, 0, 0, 0])

    _fill_valid_plan(miss_global_indices, miss_hot_indices, num_host_rows, num_blocks * block_size)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, before)


@torch.no_grad()
def test_hisparse_gather_compact_mixed_miss_counts():
    row_bytes = 32
    num_rows = 4
    top_k = 4
    num_host_rows = 16
    block_size = 4
    num_blocks = 4
    num_hot_rows = block_size * num_blocks

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    hot_cache = _make_hot_cache(num_blocks, block_size, row_bytes, fill_value=99)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k, [0, 1, 2, 4])

    for row in range(num_rows):
        for col in range(top_k):
            miss_global_indices[row, col] = row * top_k + col
            miss_hot_indices[row, col] = row * top_k + col

    expected = _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, expected)


@pytest.mark.parametrize("top_k", [1, 2, 8, 16])
@torch.no_grad()
def test_hisparse_gather_compact_various_top_k(top_k):
    row_bytes = 16
    num_rows = 4
    num_host_rows = max(num_rows * top_k, 32)
    block_size = 8
    num_blocks = max(2, (num_rows * top_k + block_size - 1) // block_size)
    num_hot_rows = num_blocks * block_size

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    hot_cache = _make_hot_cache(num_blocks, block_size, row_bytes, fill_value=33)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k)

    _fill_valid_plan(miss_global_indices, miss_hot_indices, num_host_rows, num_hot_rows)
    miss_counts.fill_(top_k)

    expected = _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, expected)


@torch.no_grad()
def test_hisparse_gather_compact_uses_absolute_hot_row_indices():
    row_bytes = 32
    num_rows = 2
    top_k = 2
    num_host_rows = 8
    block_size = 3
    num_blocks = 3
    num_hot_rows = block_size * num_blocks

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    hot_cache = _make_hot_cache(num_blocks, block_size, row_bytes, fill_value=55)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k, [2, 2])

    miss_global_indices[0] = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    miss_global_indices[1] = torch.tensor([5, 7], dtype=torch.int32, device=DEVICE)
    miss_hot_indices[0] = torch.tensor([2, 4], dtype=torch.int32, device=DEVICE)
    miss_hot_indices[1] = torch.tensor([6, 8], dtype=torch.int32, device=DEVICE)

    expected = _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, expected)


@torch.no_grad()
def test_hisparse_gather_compact_non_contiguous_block_stride():
    row_bytes = 16
    block_size = 2
    num_blocks = 3
    num_host_rows = 6
    num_rows = 2
    top_k = 2

    host_cache = _make_host_cache(num_host_rows, row_bytes)
    storage = torch.full((num_blocks * block_size * row_bytes + num_blocks,), 77, dtype=torch.uint8, device=DEVICE)
    hot_cache = torch.as_strided(storage, size=(num_blocks, block_size, row_bytes), stride=(block_size * row_bytes + 1, row_bytes, 1))

    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(num_rows, top_k, [2, 2])
    miss_global_indices[0] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    miss_global_indices[1] = torch.tensor([4, 5], dtype=torch.int32, device=DEVICE)
    miss_hot_indices[0] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    miss_hot_indices[1] = torch.tensor([4, 5], dtype=torch.int32, device=DEVICE)

    expected = _expected_after_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    _run_compact(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)

    torch.testing.assert_close(hot_cache, expected)

    for block in range(num_blocks - 1):
        padding_offset = block * (block_size * row_bytes + 1) + block_size * row_bytes
        assert storage[padding_offset].item() == 77


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.int16])
@torch.no_grad()
def test_hisparse_gather_compact_rejects_invalid_plan_dtype(bad_dtype):
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(2, 16)
    hot_cache = _make_hot_cache(1, 2, 16)
    miss_global_indices = torch.zeros((1, 1), dtype=bad_dtype, device=DEVICE)
    miss_hot_indices = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    miss_counts = torch.ones((1,), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="compact miss plan must be matching contiguous int32 CUDA tensors"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_non_pinned_host_cache():
    op = _require_hisparse_compact_op()
    host_cache = torch.ones((2, 16), dtype=torch.uint8)
    hot_cache = _make_hot_cache(1, 2, 16)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="host_cache must be pinned CPU memory"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_non_cuda_hot_cache():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(2, 16)
    hot_cache = torch.empty((1, 2, 16), dtype=torch.uint8)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="hot_cache must be on CUDA"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_non_3d_hot_cache():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(2, 16)
    hot_cache = torch.empty((2, 16), dtype=torch.uint8, device=DEVICE)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="hot_cache must be \\[num_blocks, block_size, row_width\\]"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_unaligned_row_bytes():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(2, 17)
    hot_cache = _make_hot_cache(1, 2, 17)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(1, 1)

    with pytest.raises(RuntimeError, match="KV rows must be 16-byte aligned"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_non_contiguous_plan():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(4, 16)
    hot_cache = _make_hot_cache(1, 4, 16)
    miss_global_indices = torch.zeros((1, 2), dtype=torch.int32, device=DEVICE).t()
    miss_hot_indices = torch.zeros((1, 2), dtype=torch.int32, device=DEVICE)
    miss_counts = torch.ones((1,), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="compact miss plan must be matching contiguous int32 CUDA tensors"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_mismatched_plan_shapes():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(4, 16)
    hot_cache = _make_hot_cache(1, 4, 16)
    miss_global_indices = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    miss_hot_indices = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    miss_counts = torch.ones((2,), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="compact miss plan must be matching contiguous int32 CUDA tensors"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@torch.no_grad()
def test_hisparse_gather_compact_rejects_short_miss_counts():
    op = _require_hisparse_compact_op()
    host_cache = _make_host_cache(4, 16)
    hot_cache = _make_hot_cache(1, 4, 16)
    miss_global_indices = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    miss_hot_indices = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    miss_counts = torch.ones((1,), dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="compact miss plan must be matching contiguous int32 CUDA tensors"):
        op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)


@pytest.mark.parametrize("rows,top_k", [(0, 1), (1, 0), (0, 0)])
@torch.no_grad()
def test_hisparse_gather_compact_empty_input(rows, top_k):
    op = _require_hisparse_compact_op()
    row_bytes = 16
    host_cache = torch.empty((1, row_bytes), dtype=torch.uint8).pin_memory()
    hot_cache = _make_hot_cache(1, max(top_k, 1), row_bytes)
    miss_global_indices, miss_hot_indices, miss_counts = _make_plan(rows, top_k)

    op(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts)
    torch.accelerator.synchronize()