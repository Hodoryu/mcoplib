# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for mcoplib vocab_parallel_embedding."""

import pytest
import torch
import torch.nn.functional as F

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "vocab_parallel_embedding"


def _get_op():
    for namespace_name in ("_C", "mcoplib"):
        namespace = getattr(torch.ops, namespace_name, None)
        if namespace is not None and hasattr(namespace, OP_NAME):
            return getattr(namespace, OP_NAME)
    pytest.skip(f"mcoplib operator {OP_NAME!r} is not registered")


def _pad_vocab_size(vocab_size, pad=64):
    return ((vocab_size + pad - 1) // pad) * pad


def _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size, pad=64):
    """Compute the vocabulary ranges assigned to one TP rank."""
    assert 0 < org_vocab_size <= vocab_size
    padded_vocab_size = _pad_vocab_size(vocab_size, pad)
    padded_org_vocab_size = _pad_vocab_size(org_vocab_size, pad)
    assert padded_vocab_size % tp_size == 0

    num_org_vocab_padding = padded_org_vocab_size - org_vocab_size
    rows_per_rank = padded_vocab_size // tp_size
    shard_start = tp_rank * rows_per_rank
    shard_end = shard_start + rows_per_rank

    org_start = min(shard_start, org_vocab_size)
    org_end = min(shard_end, org_vocab_size)

    # Added vocabulary starts after the padded original vocabulary.
    # A rank only owns the added-vocabulary portion intersecting its shard.
    added_start = org_vocab_size + max(0, shard_start - padded_org_vocab_size)
    added_end = org_vocab_size + max(0, shard_end - padded_org_vocab_size)
    added_start = min(added_start, vocab_size)
    added_end = min(added_end, vocab_size)

    return {
        "org_vocab_start_index": org_start,
        "org_vocab_end_index": org_end,
        "num_org_vocab_padding": num_org_vocab_padding,
        "added_vocab_start_index": added_start,
        "added_vocab_end_index": added_end,
        "num_added_vocab": max(0, added_end - added_start),
        "shard_start": shard_start,
        "shard_end": shard_end,
        "rows_per_rank": rows_per_rank,
    }


def _reference(input_ids, weight, si):
    """Reference implementation using local indices and masking."""
    ids = input_ids.long()

    org_start = si["org_vocab_start_index"]
    org_end = si["org_vocab_end_index"]
    num_org_padding = si["num_org_vocab_padding"]
    added_start = si["added_vocab_start_index"]
    added_end = si["added_vocab_end_index"]

    is_org = (ids >= org_start) & (ids < org_end)
    is_added = (ids >= added_start) & (ids < added_end)
    input_mask = ~(is_org | is_added)

    masked_input = torch.zeros_like(ids)
    masked_input = torch.where(is_org, ids - org_start, masked_input)
    masked_input = torch.where(is_added, ids - added_start + (org_end - org_start) + num_org_padding, masked_input)

    # Check that every embedding index is valid before launching embedding.
    if masked_input.numel() > 0:
        assert masked_input.min().item() >= 0
        assert masked_input.max().item() < weight.size(0)

    out = F.embedding(masked_input, weight)
    return out.masked_fill(input_mask.unsqueeze(-1), 0)


def _fused(input_ids, weight, si):
    out = torch.empty((input_ids.numel(), weight.shape[1]), dtype=weight.dtype, device=weight.device)
    _get_op()(out, input_ids, weight, si["org_vocab_start_index"], si["org_vocab_end_index"], si["num_org_vocab_padding"], si["added_vocab_start_index"], si["added_vocab_end_index"])
    return out


def _shard(full_table, si, dtype):
    """Construct the local weight using the same row layout as the operator."""
    rows_per_rank = si["rows_per_rank"]
    weight = torch.zeros((rows_per_rank, full_table.shape[1]), dtype=dtype, device=full_table.device)

    org_start = si["org_vocab_start_index"]
    org_end = si["org_vocab_end_index"]
    num_org_rows = org_end - org_start

    if num_org_rows > 0:
        weight[:num_org_rows] = full_table[org_start:org_end]

    added_start = si["added_vocab_start_index"]
    added_end = si["added_vocab_end_index"]
    num_added_rows = added_end - added_start

    if num_added_rows > 0:
        added_at = num_org_rows + si["num_org_vocab_padding"]
        added_row_end = added_at + num_added_rows
        assert added_at >= 0
        assert added_row_end <= rows_per_rank
        weight[added_at:added_row_end] = full_table[added_start:added_end]

    return weight


def _require_device():
    if not torch.cuda.is_available():
        pytest.skip("No accelerator available through torch.cuda")


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("vocab_size,org_vocab_size", [(128256, 128256), (32256, 32000)])
@pytest.mark.parametrize("hidden", [4096, 3584])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_matches_masked_gather(tp_size, vocab_size, org_vocab_size, hidden, id_dtype):
    _require_device()
    torch.manual_seed(7)

    dtype = torch.bfloat16
    padded_vocab_size = _pad_vocab_size(vocab_size, 64)
    assert padded_vocab_size % tp_size == 0
    full_table = torch.randn((padded_vocab_size, hidden), dtype=dtype, device=DEVICE)
    ids = torch.randint(0, vocab_size, (257,), dtype=id_dtype, device=DEVICE)

    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        weight = _shard(full_table, si, dtype)

        actual = _fused(ids, weight, si)
        expected = _reference(ids, weight, si)

        # Synchronize here so asynchronous device failures are reported close to their origin.
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("tp_size", [2, 8])
@torch.inference_mode()
def test_rank_sum_is_full_lookup(tp_size):
    _require_device()
    torch.manual_seed(11)

    vocab_size, org_vocab_size, hidden = 32256, 32000, 2048
    dtype = torch.bfloat16
    padded_vocab_size = _pad_vocab_size(vocab_size, 64)
    assert padded_vocab_size % tp_size == 0

    full_table = torch.randn((padded_vocab_size, hidden), dtype=dtype, device=DEVICE)
    ids = torch.randint(0, vocab_size, (129,), dtype=torch.int32, device=DEVICE)
    ids[:16] = torch.randint(org_vocab_size, vocab_size, (16,), dtype=torch.int32, device=DEVICE)

    total = torch.zeros((ids.shape[0], hidden), dtype=dtype, device=DEVICE)

    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        weight = _shard(full_table, si, dtype)
        total += _fused(ids, weight, si)

    torch.cuda.synchronize()
    torch.testing.assert_close(total, full_table[ids.long()], atol=0.0, rtol=0.0)