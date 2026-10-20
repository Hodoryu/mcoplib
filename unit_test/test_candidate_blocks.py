"""Correctness and validation tests for the vLLM candidate-block operators."""

import re
from typing import Optional

import pytest
import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="candidate-block ops require a CUDA device"
)


def _block_scores(
    logits: torch.Tensor,
    starts: Optional[torch.Tensor],
    ends: torch.Tensor,
    block_size: int,
    row_repeat: int,
) -> torch.Tensor:
    rows, width = logits.shape
    nblocks = (width + block_size - 1) // block_size
    result = torch.full(
        (rows, nblocks), -torch.inf, device=logits.device, dtype=logits.dtype
    )
    for row in range(rows):
        bound_row = row // row_repeat
        start = int(starts[bound_row]) if starts is not None else 0
        end = int(ends[bound_row])
        visible = max(0, min(end, width) - start)
        packed = torch.full(
            (nblocks * block_size,),
            -torch.inf,
            device=logits.device,
            dtype=logits.dtype,
        )
        if visible:
            packed[:visible] = logits[row, start : start + visible]
        result[row] = packed.view(nblocks, block_size).amax(dim=1)
        if end > start:
            newest = (end - start - 1) // block_size
            if newest < nblocks:
                result[row, newest] = torch.inf
    return result


def _assert_topk_set(actual: torch.Tensor, scores: torch.Tensor) -> None:
    topk = actual.size(1)
    nblocks = scores.size(1)
    for row in range(actual.size(0)):
        selected = actual[row][actual[row] >= 0].to(torch.long)
        assert bool((selected < nblocks).all())
        assert selected.unique().numel() == selected.numel()

        budget = min(topk, nblocks)
        expected_values = torch.topk(scores[row], budget).values
        expected_values = expected_values[expected_values > -torch.inf]
        actual_values = scores[row, selected]
        torch.testing.assert_close(
            actual_values.sort().values,
            expected_values.sort().values,
            rtol=0,
            atol=0,
        )
        assert int((actual[row] == -1).sum()) == topk - selected.numel()


def _mask_reference(
    logits: torch.Tensor,
    starts: Optional[torch.Tensor],
    ends: torch.Tensor,
    candidates: torch.Tensor,
    block_size: int,
    row_repeat: int,
) -> torch.Tensor:
    expected = logits.clone()
    rows, width = expected.shape
    nblocks = (width + block_size - 1) // block_size
    for row in range(rows):
        bound_row = row // row_repeat
        start = int(starts[bound_row]) if starts is not None else 0
        end = int(ends[bound_row])
        selected = set()
        edge = False
        for candidate in candidates[row].tolist():
            if candidate < 0:
                continue
            if start + candidate * block_size >= width:
                edge = True
            elif candidate < nblocks:
                selected.add(candidate)
        for col in range(width):
            keep = start <= col < end and (
                (col - start) // block_size in selected
                or (col == width - 1 and edge)
            )
            if not keep:
                expected[row, col] = -torch.inf
    return expected


@pytest.mark.parametrize(
    "rows,width,block_size,topk,starts_data,ends_data,row_repeat",
    [
        (1, 1, 8, 4, None, [1], 1),
        (2, 17, 8, 8, [3, 2], [15, 17], 1),
        (4, 63, 7, 16, [2, 9], [62, 51], 2),
        (4, 64, 8, 8, None, [24, 59], 2),
        (2, 64, 8, 16, None, [0, 61], 1),
    ],
)
def test_candidate_blocks_matches_reference(
    rows: int,
    width: int,
    block_size: int,
    topk: int,
    starts_data,
    ends_data,
    row_repeat: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        rows + width + block_size + topk
    )
    logits = torch.randn(
        (rows, width), device="cuda", dtype=torch.float32, generator=generator
    )
    starts = (
        torch.tensor(starts_data, device="cuda", dtype=torch.int32)
        if starts_data is not None
        else None
    )
    ends = torch.tensor(ends_data, device="cuda", dtype=torch.int32)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)

    torch.ops._C.select_candidate_blocks(
        logits, starts, ends, topk, block_size, out, row_repeat
    )
    scores = _block_scores(logits, starts, ends, block_size, row_repeat)
    _assert_topk_set(out, scores)

    actual = logits.clone()
    expected = _mask_reference(
        logits, starts, ends, out, block_size, row_repeat
    )
    torch.ops._C.apply_candidate_mask(
        actual, starts, ends, out, block_size, row_repeat
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_candidate_blocks_support_strided_logits() -> None:
    rows, width, block_size, topk = 2, 35, 6, 5
    storage = torch.randn((rows, width * 2), device="cuda", dtype=torch.float32)
    logits = storage[:, ::2]
    starts = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    ends = torch.tensor([31, 34], device="cuda", dtype=torch.int32)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    assert logits.stride(1) == 2

    torch.ops._C.select_candidate_blocks(
        logits, starts, ends, topk, block_size, out
    )
    _assert_topk_set(out, _block_scores(logits, starts, ends, block_size, 1))

    expected = _mask_reference(logits, starts, ends, out, block_size, 1)
    torch.ops._C.apply_candidate_mask(logits, starts, ends, out, block_size)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_apply_candidate_mask_edge_candidate_keeps_last_column() -> None:
    logits = torch.arange(17, device="cuda", dtype=torch.float32).view(1, -1)
    ends = torch.tensor([17], device="cuda", dtype=torch.int32)
    candidates = torch.tensor([[0, 3, -1]], device="cuda", dtype=torch.int32)
    expected = _mask_reference(logits, None, ends, candidates, 8, 1)

    torch.ops._C.apply_candidate_mask(logits, None, ends, candidates, 8)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert logits[0, -1].isfinite()


def test_select_candidate_blocks_zero_width() -> None:
    logits = torch.empty((3, 0), device="cuda", dtype=torch.float32)
    ends = torch.zeros(3, device="cuda", dtype=torch.int32)
    out = torch.empty((3, 4), device="cuda", dtype=torch.int32)
    torch.ops._C.select_candidate_blocks(logits, None, ends, 4, 8, out)
    assert bool((out == -1).all())


@pytest.mark.parametrize(
    "make_logits,make_ends,message",
    [
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float16),
            lambda: torch.full((2,), 8, device="cuda", dtype=torch.int32),
            "logits must be a CUDA float32 matrix",
        ),
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float32),
            lambda: torch.full((2,), 8, device="cuda", dtype=torch.int64),
            "ends must be CUDA int32 vector",
        ),
    ],
)
def test_select_candidate_blocks_validation(make_logits, make_ends, message: str) -> None:
    out = torch.empty((2, 2), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=re.escape(message)):
        torch.ops._C.select_candidate_blocks(
            make_logits(), None, make_ends(), 2, 8, out
        )

