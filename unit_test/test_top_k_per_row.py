# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

OP_NAME = "top_k_per_row_decode"
SORTING_THRESHOLD = 12288
SPLIT_WORK_THRESHOLD = 200_000


def _get_op():
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA")
    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"mcoplib operator {OP_NAME} is not registered")
    return getattr(torch.ops._C, OP_NAME)


def _make_logits(num_rows, vocab_size, seed=1234, non_contiguous=False):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if non_contiguous:
        logits = torch.randn((num_rows, vocab_size * 2), generator=generator, dtype=torch.float32).to("cuda")[:, ::2]
    else:
        logits = torch.randn((num_rows, vocab_size), generator=generator, dtype=torch.float32).to("cuda")
    return logits


def _make_seq_lens(batch_size, next_n, vocab_size, top_k=1, seed=5678):
    if vocab_size < top_k:
        raise ValueError(f"vocab_size ({vocab_size}) must be >= top_k ({top_k})")
    if vocab_size < next_n:
        raise ValueError(f"vocab_size ({vocab_size}) must be >= next_n ({next_n})")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randint(low=next_n, high=vocab_size, size=(batch_size,), dtype=torch.int32, device="cuda", generator=generator)


def _get_row_ends(seq_lens, next_n):
    num_rows = seq_lens.numel() * next_n
    row_indices = torch.arange(num_rows, device=seq_lens.device) // next_n
    next_n_offset = torch.arange(num_rows, device=seq_lens.device) % next_n
    return seq_lens[row_indices] - next_n + next_n_offset + 1


def _run_decode(logits, seq_lens, next_n, top_k):
    num_rows = logits.size(0)
    indices = torch.empty((num_rows, top_k), device=logits.device, dtype=torch.int32)
    _get_op()(logits, next_n, seq_lens, indices, num_rows, logits.stride(0), logits.stride(1), top_k)
    torch.cuda.synchronize()
    return indices


def _check_indices(indices, row_ends, top_k):
    assert indices.dtype == torch.int32
    assert indices.shape[1] == top_k
    for row in range(indices.size(0)):
        valid_len = min(int(row_ends[row].item()), top_k)
        row_indices = indices[row, :valid_len]
        assert torch.all(row_indices >= 0).item(), f"Row {row}: negative index: {row_indices}"
        assert torch.all(row_indices < int(row_ends[row].item())).item(), f"Row {row}: index exceeds row length: {row_indices}"
        if valid_len > 1:
            sorted_indices = row_indices.sort().values
            assert torch.all(sorted_indices[1:] != sorted_indices[:-1]).item(), f"Row {row}: duplicate indices: {row_indices}"


def _check_exact_topk(logits, indices, row_ends, top_k):
    for row in range(logits.size(0)):
        row_end = min(int(row_ends[row].item()), logits.size(1))
        k_i = min(top_k, row_end)
        assert k_i > 0, f"Row {row}: invalid row_end={row_end}"
        expected = torch.topk(logits[row, :row_end], k_i, largest=True, sorted=True).values
        actual_indices = indices[row, :k_i].to(torch.int64)
        actual = logits[row].gather(0, actual_indices)
        actual = torch.sort(actual, descending=True).values
        assert torch.allclose(actual, expected, rtol=0, atol=0), (
            f"Top-K mismatch at row {row}: row_end={row_end}, top_k={top_k}, "
            f"actual={actual.tolist()}, expected={expected.tolist()}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("next_n", [1, 2, 4, 8])
@pytest.mark.parametrize("vocab_size", [1024, 4096, 12287])
@pytest.mark.parametrize("top_k", [1, 8, 32])
def test_top_k_per_row_decode_basic(batch_size, next_n, vocab_size, top_k):
    if top_k > vocab_size:
        pytest.skip("top_k must not exceed vocab_size")
    num_rows = batch_size * next_n
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k)
    row_ends = _get_row_ends(seq_lens, next_n)
    if torch.any(row_ends < top_k).item():
        pytest.skip("Some row lengths are smaller than top_k")
    logits = _make_logits(num_rows, vocab_size)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(indices, row_ends, top_k)
    _check_exact_topk(logits, indices, row_ends, top_k)


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("top_k", [1, 8, 32])
@pytest.mark.parametrize("vocab_size", [12287, 12288, 12289, 199999, 200000, 200001])
def test_top_k_per_row_decode_algorithm_boundaries(next_n, top_k, vocab_size):
    batch_size = 2
    num_rows = batch_size * next_n
    seq_lens = torch.full((batch_size,), vocab_size - 1, dtype=torch.int32, device="cuda")
    row_ends = _get_row_ends(seq_lens, next_n)
    if torch.any(row_ends < top_k).item():
        pytest.skip("Some row lengths are smaller than top_k")
    logits = _make_logits(num_rows, vocab_size, seed=2025)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(indices, row_ends, top_k)
    _check_exact_topk(logits, indices, row_ends, top_k)


@pytest.mark.parametrize("next_n", [1, 2, 4, 8])
@pytest.mark.parametrize("top_k", [1, 8, 32])
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_top_k_per_row_decode_non_contiguous_logits(next_n, top_k, non_contiguous):
    batch_size = 2
    vocab_size = 4096
    num_rows = batch_size * next_n
    seq_lens = torch.full((batch_size,), vocab_size - 1, dtype=torch.int32, device="cuda")
    row_ends = _get_row_ends(seq_lens, next_n)
    logits = _make_logits(num_rows, vocab_size, seed=2026, non_contiguous=non_contiguous)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(indices, row_ends, top_k)
    _check_exact_topk(logits, indices, row_ends, top_k)


@pytest.mark.parametrize("next_n", [1, 2, 4])
@pytest.mark.parametrize("top_k", [8, 32])
def test_top_k_per_row_decode_ties(next_n, top_k):
    batch_size = 2
    vocab_size = 4096
    num_rows = batch_size * next_n
    seq_lens = torch.full((batch_size,), vocab_size - 1, dtype=torch.int32, device="cuda")
    row_ends = _get_row_ends(seq_lens, next_n)
    logits = torch.zeros((num_rows, vocab_size), dtype=torch.float32, device="cuda")
    logits[:, :top_k * 2] = 1.0
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(indices, row_ends, top_k)
    for row in range(num_rows):
        selected = logits[row].gather(0, indices[row].to(torch.int64))
        assert torch.all(selected == 1.0).item(), f"Row {row}: selected non-top value"


@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_top_k_per_row_decode_repeatability(next_n):
    batch_size = 2
    vocab_size = 4096
    top_k = 32
    num_rows = batch_size * next_n
    seq_lens = torch.full((batch_size,), vocab_size - 1, dtype=torch.int32, device="cuda")
    row_ends = _get_row_ends(seq_lens, next_n)
    logits = _make_logits(num_rows, vocab_size, seed=2026)
    indices_first = _run_decode(logits, seq_lens, next_n, top_k)
    indices_second = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(indices_first, row_ends, top_k)
    _check_indices(indices_second, row_ends, top_k)
    _check_exact_topk(logits, indices_first, row_ends, top_k)
    _check_exact_topk(logits, indices_second, row_ends, top_k)
    assert torch.equal(indices_first, indices_second), "Results differ between repeated calls"