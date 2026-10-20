import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Selective_scan_fwd_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.batch = config.get("batch", config.get("batch_size", 2))
        self.dim = config.get("dim", 64)
        self.seqlen = config.get("seqlen", config.get("seq_len", 128))
        self.dstate = config.get("dstate", 16)
        self.n_groups = config.get("n_groups", config.get("varBC_groups", 1))
        self.delta_softplus = config.get("delta_softplus", True)
        self.pad_slot_id = config.get("pad_slot_id", -1)

        self.has_delta_bias = config.get("has_delta_bias", True)
        self.has_z = config.get("has_z", True)
        self.has_D = config.get("has_D", True)

        self.dtype = self._parse_dtype(config.get("dtype", self.dtype))

        if self.batch <= 0 or self.dim <= 0 or self.seqlen <= 0 or self.dstate <= 0:
            raise ValueError("batch、dim、seqlen 和 dstate 必须大于 0")

        if self.n_groups <= 0 or self.dim % self.n_groups != 0:
            raise ValueError(f"dim={self.dim} 必须能被 n_groups={self.n_groups} 整除")

        self.shape = (self.batch, self.dim, self.seqlen)

    @staticmethod
    def _parse_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype

        dtype_map = {
            "float": torch.float32,
            "float32": torch.float32,
            "torch.float32": torch.float32,
            "half": torch.float16,
            "float16": torch.float16,
            "torch.float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
        }

        key = str(dtype).lower()
        if key not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[key]

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B={self.batch} D={self.dim} L={self.seqlen})")
        state.add_summary("dstate", str(self.dstate))
        state.add_summary("n_groups", str(self.n_groups))

        total = self.batch * self.dim * self.seqlen
        state.add_element_count(total)

        element_size = torch.empty((), dtype=self.dtype).element_size()
        weight_size = torch.empty((), dtype=torch.float32).element_size()

        bc_elements = 2 * self.batch * self.n_groups * self.dstate * self.seqlen
        read_bytes = 2 * total * element_size + bc_elements * element_size + self.dim * self.dstate * weight_size

        if self.has_D:
            read_bytes += self.dim * weight_size
        if self.has_delta_bias:
            read_bytes += self.dim * weight_size
        if self.has_z:
            read_bytes += total * element_size

        write_bytes = total * element_size + self.batch * self.dim * self.dstate * element_size

        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _make_inputs(self, dev):
        u = torch.randn((self.batch, self.dim, self.seqlen), dtype=self.dtype, device=dev)
        delta = torch.randn_like(u)
        A = -torch.rand((self.dim, self.dstate), dtype=torch.float32, device=dev)

        if self.n_groups == 1:
            B = torch.randn((self.batch, self.dstate, self.seqlen), dtype=self.dtype, device=dev)
            C = torch.randn_like(B)
        else:
            B = torch.randn((self.batch, self.n_groups, self.dstate, self.seqlen), dtype=self.dtype, device=dev)
            C = torch.randn_like(B)

        D = torch.randn(self.dim, dtype=torch.float32, device=dev) if self.has_D else None
        z = torch.randn_like(u) if self.has_z else None
        delta_bias = torch.randn(self.dim, dtype=torch.float32, device=dev) if self.has_delta_bias else None
        ssm_states = torch.zeros((self.batch, self.dim, self.dstate), dtype=self.dtype, device=dev)

        return u, delta, A, B, C, D, z, delta_bias, ssm_states

    def _call_op(self, inputs):
        u, delta, A, B, C, D, z, delta_bias, ssm_states = inputs

        if B.dim() == 3:
            B = B.unsqueeze(1).contiguous()
            C = C.unsqueeze(1).contiguous()

        torch.ops._C.selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, self.delta_softplus, None, None, None, ssm_states, self.pad_slot_id, 1, None, None, None, None, None)

        return z if z is not None else delta

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            inputs = self._make_inputs(dev)
            inputs_copy = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)

        def op_closure():
            current_inputs = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs_copy)
            self._call_op(current_inputs)

        return self.make_launcher(dev_id, op_closure)

    def selective_scan_ref(self, u, delta, A, B, C, D, z, delta_bias, ssm_states):
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]

        u_fp32 = u.float()
        delta_fp32 = delta.float()
        A_fp32 = A.float()
        B_fp32 = B.float()
        C_fp32 = C.float()
        state = ssm_states.float().clone()

        if delta_bias is not None:
            delta_fp32 = delta_fp32 + delta_bias.float().view(1, dim, 1)

        if self.delta_softplus:
            delta_fp32 = F.softplus(delta_fp32)

        if B_fp32.dim() == 3:
            B_fp32 = B_fp32.unsqueeze(1)

        if B_fp32.shape[1] == 1:
            B_fp32 = B_fp32.expand(batch, dim, dstate, seqlen)
            C_fp32 = C_fp32.expand(batch, dim, dstate, seqlen)
        else:
            group_size = dim // B_fp32.shape[1]
            B_fp32 = B_fp32.repeat_interleave(group_size, dim=1)
            C_fp32 = C_fp32.repeat_interleave(group_size, dim=1)

        outputs = []

        for i in range(seqlen):
            u_i = u_fp32[:, :, i]
            delta_i = delta_fp32[:, :, i]
            B_i = B_fp32[:, :, :, i]
            C_i = C_fp32[:, :, :, i]

            delta_A = torch.exp(delta_i.unsqueeze(-1) * A_fp32.unsqueeze(0))
            delta_B_u = delta_i.unsqueeze(-1) * B_i * u_i.unsqueeze(-1)

            state = delta_A * state + delta_B_u
            y = (state * C_i).sum(dim=-1)

            if D is not None:
                y = y + D.float().view(1, dim) * u_i

            outputs.append(y)

        out = torch.stack(outputs, dim=-1)

        if z is not None:
            out = out * F.silu(z.float())

        return out.to(u.dtype), state

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        torch.manual_seed(0)

        inputs = self._make_inputs(dev)
        u, delta, A, B, C, D, z, delta_bias, ssm_states = inputs

        u_ref = u.clone()
        delta_ref = delta.clone()
        A_ref = A.clone()
        B_ref = B.clone()
        C_ref = C.clone()
        D_ref = D.clone() if D is not None else None
        z_ref = z.clone() if z is not None else None
        delta_bias_ref = delta_bias.clone() if delta_bias is not None else None
        state_ref_input = ssm_states.clone()

        op_inputs = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
        op_out = self._call_op(op_inputs)

        ref_out, _ = self.selective_scan_ref(u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z_ref, delta_bias_ref, state_ref_input)

        return self.check_diff(op_out, ref_out.to(op_out.dtype), 1e-2)