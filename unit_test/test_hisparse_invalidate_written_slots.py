import pytest
import torch
import mcoplib._C


OP = torch.ops._C.hisparse_invalidate_written_slots
DEVICE = "cuda"


def _base_inputs(num_states=4, region_stride=8, num_tokens=4, num_requests=4):
    device_global_indices = torch.arange(num_states * region_stride, dtype=torch.int32, device=DEVICE).reshape(num_states, region_stride).contiguous()
    request_state_indices = torch.arange(num_requests, dtype=torch.int32, device=DEVICE).contiguous()
    req_id_per_token = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE).remainder(max(num_requests, 1)).contiguous()
    written_slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE).remainder(max(region_stride, 1)).contiguous()
    return {"device_global_indices": device_global_indices, "request_state_indices": request_state_indices, "req_id_per_token": req_id_per_token, "written_slots": written_slots}


def _invoke(args):
    OP(args["device_global_indices"], args["request_state_indices"], args["req_id_per_token"], args["written_slots"])


def _assert_raises(args, message):
    with pytest.raises(RuntimeError, match=message):
        _invoke(args)


# ---------------------------------------------------------------------------
# Basic invocation and empty inputs
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_basic():
    args = _base_inputs()
    _invoke(args)
    torch.cuda.synchronize()


@pytest.mark.parametrize("num_states,region_stride,num_tokens,num_requests", [(0, 8, 4, 4), (4, 0, 4, 4), (4, 8, 0, 4), (0, 0, 0, 0)])
def test_hisparse_invalidate_written_slots_empty_inputs(num_states, region_stride, num_tokens, num_requests):
    args = _base_inputs(num_states=num_states, region_stride=region_stride, num_tokens=num_tokens, num_requests=num_requests)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_invalidate_written_slots_single_token():
    args = _base_inputs(num_states=1, region_stride=1, num_tokens=1, num_requests=1)
    _invoke(args)
    torch.cuda.synchronize()


@pytest.mark.parametrize("num_states,region_stride,num_tokens,num_requests", [(1, 16, 16, 1), (2, 32, 8, 2), (8, 64, 32, 8), (16, 128, 64, 16)])
def test_hisparse_invalidate_written_slots_various_shapes(num_states, region_stride, num_tokens, num_requests):
    args = _base_inputs(num_states=num_states, region_stride=region_stride, num_tokens=num_tokens, num_requests=num_requests)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_invalidate_written_slots_accepts_both_empty_token_arrays():
    args = _base_inputs(num_tokens=0)
    args["req_id_per_token"] = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    args["written_slots"] = torch.empty((0,), dtype=torch.int64, device=DEVICE)
    _invoke(args)


# ---------------------------------------------------------------------------
# CUDA device validation
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_device_global_indices_on_cpu():
    args = _base_inputs()
    args["device_global_indices"] = args["device_global_indices"].cpu()
    _assert_raises(args, "HiSparse invalidation tensors must be on CUDA")


def test_hisparse_invalidate_written_slots_rejects_request_state_indices_on_cpu():
    args = _base_inputs()
    args["request_state_indices"] = args["request_state_indices"].cpu()
    _assert_raises(args, "HiSparse invalidation tensors must be on CUDA")


def test_hisparse_invalidate_written_slots_rejects_req_id_per_token_on_cpu():
    args = _base_inputs()
    args["req_id_per_token"] = args["req_id_per_token"].cpu()
    _assert_raises(args, "HiSparse invalidation tensors must be on CUDA")


def test_hisparse_invalidate_written_slots_rejects_written_slots_on_cpu():
    args = _base_inputs()
    args["written_slots"] = args["written_slots"].cpu()
    _assert_raises(args, "HiSparse invalidation tensors must be on CUDA")


# ---------------------------------------------------------------------------
# device_global_indices validation
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_device_global_indices_wrong_dtype():
    args = _base_inputs()
    args["device_global_indices"] = args["device_global_indices"].to(torch.int64)
    _assert_raises(args, "device_global_indices must be a contiguous 2D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_device_global_indices_wrong_dim():
    args = _base_inputs()
    args["device_global_indices"] = torch.zeros((32,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "device_global_indices must be a contiguous 2D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_device_global_indices_noncontiguous():
    args = _base_inputs()
    args["device_global_indices"] = torch.zeros((4, 16), dtype=torch.int32, device=DEVICE)[:, ::2]
    _assert_raises(args, "device_global_indices must be a contiguous 2D int32 CUDA tensor")


# ---------------------------------------------------------------------------
# request_state_indices validation
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_request_state_indices_wrong_dtype():
    args = _base_inputs()
    args["request_state_indices"] = args["request_state_indices"].to(torch.int64)
    _assert_raises(args, "request_state_indices must be a contiguous 1D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_request_state_indices_wrong_dim():
    args = _base_inputs()
    args["request_state_indices"] = torch.zeros((4, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_state_indices must be a contiguous 1D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_request_state_indices_noncontiguous():
    args = _base_inputs()
    args["request_state_indices"] = torch.zeros((8,), dtype=torch.int32, device=DEVICE)[::2]
    _assert_raises(args, "request_state_indices must be a contiguous 1D int32 CUDA tensor")


# ---------------------------------------------------------------------------
# req_id_per_token validation
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_req_id_per_token_wrong_dtype():
    args = _base_inputs()
    args["req_id_per_token"] = args["req_id_per_token"].to(torch.int64)
    _assert_raises(args, "req_id_per_token must be a contiguous 1D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_req_id_per_token_wrong_dim():
    args = _base_inputs()
    args["req_id_per_token"] = torch.zeros((4, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "req_id_per_token must be a contiguous 1D int32 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_req_id_per_token_noncontiguous():
    args = _base_inputs()
    args["req_id_per_token"] = torch.zeros((8,), dtype=torch.int32, device=DEVICE)[::2]
    _assert_raises(args, "req_id_per_token must be a contiguous 1D int32 CUDA tensor")


# ---------------------------------------------------------------------------
# written_slots validation
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_written_slots_wrong_dtype():
    args = _base_inputs()
    args["written_slots"] = args["written_slots"].to(torch.int32)
    _assert_raises(args, "written_slots must be a contiguous 1D int64 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_written_slots_wrong_dim():
    args = _base_inputs()
    args["written_slots"] = torch.zeros((4, 1), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "written_slots must be a contiguous 1D int64 CUDA tensor")


def test_hisparse_invalidate_written_slots_rejects_written_slots_noncontiguous():
    args = _base_inputs()
    args["written_slots"] = torch.zeros((8,), dtype=torch.int64, device=DEVICE)[::2]
    _assert_raises(args, "written_slots must be a contiguous 1D int64 CUDA tensor")


# ---------------------------------------------------------------------------
# Length consistency
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_rejects_unequal_token_lengths():
    args = _base_inputs(num_tokens=4)
    args["written_slots"] = torch.zeros((3,), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "req_id_per_token and written_slots must have equal length")


def test_hisparse_invalidate_written_slots_rejects_empty_req_ids_with_nonempty_slots():
    args = _base_inputs(num_tokens=1)
    args["req_id_per_token"] = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "req_id_per_token and written_slots must have equal length")


# ---------------------------------------------------------------------------
# Functional correctness
# ---------------------------------------------------------------------------

def test_hisparse_invalidate_written_slots_invalidates_matching_values():
    device_global_indices = torch.tensor([[1, 2, 3, 2, 5, 2], [4, 5, 6, 5, 8, 9], [7, 8, 9, 8, 1, 8]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([1, 0, 2], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([5, 2, 8], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[1, -1, 3, -1, 5, -1], [4, -1, 6, -1, 8, 9], [7, -1, 9, -1, 1, -1]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_preserves_nonmatching_values():
    device_global_indices = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32, device=DEVICE)
    expected = device_global_indices.clone()
    request_state_indices = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([99, 100], dtype=torch.int64, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_invalidates_all_duplicates():
    device_global_indices = torch.tensor([[3, 3, 1, 3, 2, 3]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([3], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[-1, -1, 1, -1, 2, -1]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_does_not_modify_other_state_rows():
    device_global_indices = torch.tensor([[7, 7, 1, 2], [7, 3, 7, 4], [5, 7, 6, 7]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([7], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[7, 7, 1, 2], [7, 3, 7, 4], [5, -1, 6, -1]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


@pytest.mark.parametrize("req_idx,state_idx,written_slot", [(-1, 0, 3), (2, 0, 3), (0, -1, 3), (0, 2, 3), (0, 0, -1)])
def test_hisparse_invalidate_written_slots_ignores_invalid_mapping(req_idx, state_idx, written_slot):
    device_global_indices = torch.tensor([[3, 1, 3], [3, 2, 3]], dtype=torch.int32, device=DEVICE)
    expected = device_global_indices.clone()
    request_state_indices = torch.tensor([state_idx], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([req_idx], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([written_slot], dtype=torch.int64, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_repeated_tokens_same_state():
    device_global_indices = torch.tensor([[1, 2, 3, 2, 4, 3], [1, 2, 3, 4, 5, 6]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0, 0, 0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([2, 3, 4], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[1, -1, -1, -1, -1, -1], [1, 2, 3, 4, 5, 6]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_does_not_match_existing_minus_one():
    device_global_indices = torch.tensor([[-1, 2, -1, 3]], dtype=torch.int32, device=DEVICE)
    expected = device_global_indices.clone()
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([-1], dtype=torch.int64, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_zero_slot():
    device_global_indices = torch.tensor([[0, 1, 0, 2, 0]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[-1, 1, -1, 2, -1]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_preserves_unreferenced_requests():
    device_global_indices = torch.tensor([[5, 1, 5], [5, 2, 5], [5, 3, 5]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([5], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[5, 1, 5], [-1, 2, -1], [5, 3, 5]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_slot_above_region_stride():
    device_global_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=DEVICE)
    expected = device_global_indices.clone()
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([100], dtype=torch.int64, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)


def test_hisparse_invalidate_written_slots_nonmonotonic_slots():
    device_global_indices = torch.tensor([[7, 0, 5, 2, 1, 6, 3, 4]], dtype=torch.int32, device=DEVICE)
    request_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    req_id_per_token = torch.tensor([0, 0, 0], dtype=torch.int32, device=DEVICE)
    written_slots = torch.tensor([7, 2, 6], dtype=torch.int64, device=DEVICE)
    expected = torch.tensor([[-1, 0, 5, -1, 1, -1, 3, 4]], dtype=torch.int32, device=DEVICE)
    OP(device_global_indices, request_state_indices, req_id_per_token, written_slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(device_global_indices, expected)