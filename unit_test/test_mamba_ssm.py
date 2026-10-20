# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import mcoplib._C


DEVICE = "cuda"
NULL_BLOCK_ID = 0

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="selective_scan_fwd requires a CUDA-compatible device")


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, prev_state=None, final_state_out=None):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()

    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()

    x = A.new_zeros((batch, dim, dstate)) if prev_state is None else prev_state.float().clone()
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        if not is_variable_C:
            y = (x * C.unsqueeze(0)).sum(dim=-1)
        elif C.dim() == 3:
            y = (x * C[:, None, :, i]).sum(dim=-1)
        else:
            y = (x * C[:, :, :, i]).sum(dim=-1)

        if i == u.shape[2] - 1:
            if final_state_out is None:
                final_state_out = x
            else:
                final_state_out.copy_(x)

        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D.float(), "d -> d 1")

    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, final_state_out)


def selective_scan_fn_mcoplib(u, ssm_states, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, query_start_loc=None, cache_indices=None, has_initial_state=None, null_block_id=NULL_BLOCK_ID, block_size=2048, block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None, initial_state_idx=None, cu_chunk_seqlen=None, last_chunk_indices=None):
    if u.stride(-1) != 1:
        u = u.contiguous()

    if delta.stride(-1) != 1:
        delta = delta.contiguous()

    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()

    if B.stride(-1) != 1:
        B = B.contiguous()

    if C.stride(-1) != 1:
        C = C.contiguous()

    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()

    if B.dim() == 3 and query_start_loc is None:
        B = B.unsqueeze(1)

    if B.dim() == 2 and query_start_loc is not None:
        B = B.unsqueeze(0)

    if C.dim() == 3 and query_start_loc is None:
        C = C.unsqueeze(1)

    if C.dim() == 2 and query_start_loc is not None:
        C = C.unsqueeze(0)

    torch.ops._C.selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, query_start_loc, cache_indices, has_initial_state, ssm_states, null_block_id, block_size, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, cu_chunk_seqlen, last_chunk_indices)

    return delta if z is None else z


@pytest.mark.parametrize("wtype", [torch.float32])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [128, 1024, 4096])
@pytest.mark.parametrize("has_delta_bias", [True])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("has_D", [True, False])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("scan_chunks", [1, 3])
def test_selective_scan_fwd(scan_chunks, varBC_groups, has_D, has_z, has_delta_bias, delta_softplus, seqlen, itype, wtype):
    device = DEVICE

    if itype == torch.float32:
        rtol, atol = 6e-4, 2e-3
    else:
        rtol, atol = 3e-2, 5e-2

    torch.manual_seed(0)

    batch_size = 1
    dim = 4
    dstate = 8

    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    A_ref = A.clone()

    if varBC_groups == 1:
        B_shape = [batch_size, dstate, seqlen]
        C_shape = [batch_size, dstate, seqlen]
    else:
        B_shape = [batch_size, varBC_groups, dstate, seqlen]
        C_shape = [batch_size, varBC_groups, dstate, seqlen]

    B = torch.randn(B_shape, device=device, dtype=itype)
    B_ref = B.clone()

    # 诊断配置：只保留 C 的第 0 个状态分量。
    C = torch.zeros(C_shape, device=device, dtype=itype)

    if C.dim() == 3:
        C[:, 0, :] = 1.0
    else:
        C[:, :, 0, :] = 1.0

    C_ref = C.clone()

    D = torch.randn(dim, device=device, dtype=torch.float32) if has_D else None
    D_ref = D.clone() if D is not None else None

    z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype) if has_z else None
    z_ref = z.clone() if z is not None else None

    delta_bias = 0.5 * torch.rand(dim, device=device, dtype=torch.float32) if has_delta_bias else None
    delta_bias_ref = delta_bias.clone() if delta_bias is not None else None

    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    u_ref = u.clone()

    delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    delta_ref = delta.clone()

    state_shape = (batch_size, dim, dstate)
    state = torch.randn(state_shape, device=device, dtype=itype, requires_grad=False)
    state_init = state.clone()

    outs = []

    for c in range(scan_chunks):
        chunked_prompt_len = seqlen // scan_chunks
        chunk_start = chunked_prompt_len * c
        chunk_end = seqlen if c == scan_chunks - 1 else chunked_prompt_len * (c + 1)

        _u = u[..., chunk_start:chunk_end]
        _delta = delta[..., chunk_start:chunk_end]
        _B = B[..., chunk_start:chunk_end]
        _C = C[..., chunk_start:chunk_end]
        _z = z[..., chunk_start:chunk_end] if z is not None else None

        out_chunk = selective_scan_fn_mcoplib(_u, state, _delta, A, _B, _C, D=D, z=_z, delta_bias=delta_bias, delta_softplus=delta_softplus, has_initial_state=torch.ones(batch_size, device=device, dtype=torch.bool) if c > 0 else None, null_block_id=NULL_BLOCK_ID, block_size=2048)
        outs.append(out_chunk)

    out = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]

    # 保存 D 开启时 Kernel 写入 delta 的门控前输出。
    delta_with_D = delta.detach().clone()

    out_ref, state_ref = selective_scan_ref(u_ref, delta_ref, A_ref, B_ref, C_ref, D=D_ref, z=z_ref, delta_bias=delta_bias_ref, delta_softplus=delta_softplus, return_last_state=True)

    pre_z_ref, _ = selective_scan_ref(u_ref, delta_ref, A_ref, B_ref, C_ref, D=D_ref, z=None, delta_bias=delta_bias_ref, delta_softplus=delta_softplus, return_last_state=True)

    state_diff = (state.float() - state_ref.float()).abs()
    pre_z_diff = (delta_with_D.float() - pre_z_ref.float()).abs()
    out_diff = (out.float() - out_ref.float()).abs()

    print(f"seqlen={seqlen}, scan_chunks={scan_chunks}, varBC_groups={varBC_groups}, has_D={has_D}")
    print(f"State max abs diff: {state_diff.max().item():.8f}")
    print(f"Pre-z max abs diff: {pre_z_diff.max().item():.8f}")
    print(f"Final output max abs diff: {out_diff.max().item():.8f}")

    allowed = atol + rtol * out_ref.float().abs()
    bad = out_diff > allowed
    print(f"Output mismatch elements: {bad.sum().item()}/{bad.numel()}")

    # 定位当前仍然失败的 seqlen=1024、单组 B/C、FP32 场景。
    if has_D and itype == torch.float32 and seqlen == 1024 and varBC_groups == 1:
        delta_no_D = delta_ref.clone()
        z_no_D = z_ref.clone() if z_ref is not None else None
        state_no_D = state_init.clone()

        for c in range(scan_chunks):
            chunked_prompt_len = seqlen // scan_chunks
            chunk_start = chunked_prompt_len * c
            chunk_end = seqlen if c == scan_chunks - 1 else chunked_prompt_len * (c + 1)

            _u = u_ref[..., chunk_start:chunk_end]
            _delta = delta_no_D[..., chunk_start:chunk_end]
            _B = B_ref[..., chunk_start:chunk_end]
            _C = C_ref[..., chunk_start:chunk_end]
            _z = z_no_D[..., chunk_start:chunk_end] if z_no_D is not None else None

            selective_scan_fn_mcoplib(_u, state_no_D, _delta, A_ref, _B, _C, D=None, z=_z, delta_bias=delta_bias_ref, delta_softplus=delta_softplus, has_initial_state=torch.ones(batch_size, device=device, dtype=torch.bool) if c > 0 else None, null_block_id=NULL_BLOCK_ID, block_size=2048)

        # 比较 D 开启与关闭时的真实 Kernel 输出差分。
        d_term = u_ref.float() * D_ref.float().view(1, dim, 1)
        actual_d_term = delta_with_D.float() - delta_no_D.float()
        d_residual = actual_d_term - d_term

        idx_d = tuple(x.item() for x in torch.unravel_index(d_residual.abs().argmax(), d_residual.shape))
        idx_ref = tuple(x.item() for x in torch.unravel_index(pre_z_diff.argmax(), pre_z_diff.shape))

        print(f"D differential max abs diff: {d_residual.abs().max().item():.8f}")
        print(f"D differential worst index: {idx_d}")
        print(f"Kernel with D at worst D index: {delta_with_D[idx_d].item():.8f}")
        print(f"Kernel without D at worst D index: {delta_no_D[idx_d].item():.8f}")
        print(f"Expected D*u at worst D index: {d_term[idx_d].item():.8f}")
        print(f"D residual at worst index: {d_residual[idx_d].item():.8f}")
        print(f"Reference comparison worst index: {idx_ref}")
        print(f"Kernel pre-z at reference index: {delta_with_D[idx_ref].item():.8f}")
        print(f"Reference pre-z at reference index: {pre_z_ref[idx_ref].item():.8f}")

        # 比较 D=None 时每个时间步的门控前输出。
        ref_pre_no_D, _ = selective_scan_ref(u_ref, delta_ref, A_ref, B_ref, C_ref, D=None, z=None, delta_bias=delta_bias_ref, delta_softplus=delta_softplus, return_last_state=True)

        actual_no_D = delta_no_D.float()
        expected_no_D = ref_pre_no_D.float()
        err = actual_no_D - expected_no_D
        abs_err = err.abs()

        idx_no_D = tuple(x.item() for x in torch.unravel_index(abs_err.argmax(), abs_err.shape))

        print(f"D=None output max abs diff: {abs_err.max().item():.8f}")
        print(f"D=None worst index: {idx_no_D}")
        print(f"D=None kernel value: {actual_no_D[idx_no_D].item():.8f}")
        print(f"D=None reference value: {expected_no_D[idx_no_D].item():.8f}")

        for d in range(dim):
            bad_t = torch.nonzero(abs_err[0, d] > 1e-4, as_tuple=False).flatten()
            first_t = int(bad_t[0].item()) if bad_t.numel() > 0 else -1
            print(f"channel={d}, first_t_with_err_gt_1e-4={first_t}")

        for t in range(28, 41):
            print(f"t={t}, kernel={actual_no_D[0, 0, t].item():.8f}, ref={expected_no_D[0, 0, t].item():.8f}, err={err[0, 0, t].item():.8f}")

    # 状态和最终输出分别校验。
    assert torch.allclose(state, state_ref.to(state.dtype), rtol=rtol, atol=atol), f"State mismatch: max_abs_diff={state_diff.max().item()}"
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), f"Output mismatch: max_abs_diff={out_diff.max().item()}"