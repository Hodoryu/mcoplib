import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _mask_reference(logits, starts, ends, candidates, block_size, row_repeat):
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


class Apply_candidate_mask_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = int(config.get("num_rows", 8))
        self.width = int(config.get("width", 524288))
        self.topk_blocks = int(config.get("topk_blocks", 2048))
        self.block_size = int(config.get("block_size", 8))
        self.row_repeat = int(config.get("row_repeat", 1))
        if self.dtype != torch.float32:
            raise ValueError("apply_candidate_mask only supports dtype=float32")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32/int32")
        state.add_summary(
            "Shape",
            f"({self.num_rows}x{self.width} topk={self.topk_blocks} "
            f"block={self.block_size})",
        )
        logits_elements = self.num_rows * self.width
        candidate_elements = self.num_rows * self.topk_blocks
        bound_rows = (self.num_rows + self.row_repeat - 1) // self.row_repeat
        state.add_element_count(logits_elements)
        state.add_global_memory_reads(
            (logits_elements + candidate_elements + bound_rows) * 4
        )
        state.add_global_memory_writes(logits_elements * 4)

    def _prepare(self, dev_id, num_rows=None, width=None, topk_blocks=None):
        num_rows = self.num_rows if num_rows is None else num_rows
        width = self.width if width is None else width
        topk_blocks = self.topk_blocks if topk_blocks is None else topk_blocks
        device = f"cuda:{dev_id}"
        generator = torch.Generator(device=device).manual_seed(20261012)
        logits = torch.randn(
            (num_rows, width),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        bound_rows = (num_rows + self.row_repeat - 1) // self.row_repeat
        ends = torch.full((bound_rows,), width, device=device, dtype=torch.int32)
        candidates = torch.empty(
            (num_rows, topk_blocks), device=device, dtype=torch.int32
        )
        return logits, ends, candidates

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            logits, ends, candidates = self._prepare(dev_id)
            torch.ops._C.select_candidate_blocks(
                logits,
                None,
                ends,
                self.topk_blocks,
                self.block_size,
                candidates,
                self.row_repeat,
            )

        # The mask is idempotent: selected values survive every invocation and
        # all other values remain -inf. Reusing it avoids timing a reset copy.
        return self.make_launcher(
            dev_id,
            torch.ops._C.apply_candidate_mask,
            logits,
            None,
            ends,
            candidates,
            self.block_size,
            self.row_repeat,
        )

    def run_verification(self, dev_id):
        rows, width = 4, 513
        logits, ends, candidates = self._prepare(dev_id, rows, width, 8)
        ends[:] = torch.tensor(
            [0, 255, width - 3, width], device=logits.device, dtype=torch.int32
        )
        nblocks = (width + self.block_size - 1) // self.block_size
        candidates[:] = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 2, 7, 15, 23, 31, -1],
                [1, 3, 5, 17, 29, 41, 63, -1],
                [0, 2, 4, 8, 16, 32, nblocks - 1, nblocks],
            ],
            device=logits.device,
            dtype=torch.int32,
        )
        expected = _mask_reference(
            logits, None, ends, candidates, self.block_size, self.row_repeat
        )
        torch.ops._C.apply_candidate_mask(
            logits,
            None,
            ends,
            candidates,
            self.block_size,
            self.row_repeat,
        )
        torch.cuda.synchronize(dev_id)
        passed = torch.equal(logits, expected)
        return bool(passed), 0.0 if passed else 1.0
