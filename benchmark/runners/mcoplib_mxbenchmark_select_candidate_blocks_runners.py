import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _block_scores(logits, starts, ends, block_size, row_repeat):
    rows, width = logits.shape
    nblocks = (width + block_size - 1) // block_size
    scores = torch.full(
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
        scores[row] = packed.view(nblocks, block_size).amax(dim=1)
        if end > start:
            newest = (end - start - 1) // block_size
            if newest < nblocks:
                scores[row, newest] = torch.inf
    return scores


def _matches_topk(actual, scores):
    topk = actual.size(1)
    nblocks = scores.size(1)
    for row in range(actual.size(0)):
        selected = actual[row][actual[row] >= 0].to(torch.long)
        if selected.numel() and not bool((selected < nblocks).all()):
            return False
        if selected.unique().numel() != selected.numel():
            return False

        budget = min(topk, nblocks)
        expected_values = torch.topk(scores[row], budget).values
        expected_values = expected_values[expected_values > -torch.inf]
        if int((actual[row] == -1).sum()) != topk - selected.numel():
            return False
        actual_values = scores[row, selected]
        if not torch.equal(
            actual_values.sort().values, expected_values.sort().values
        ):
            return False
    return True


class Select_candidate_blocks_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = int(config.get("num_rows", 8))
        self.width = int(config.get("width", 524288))
        self.topk_blocks = int(config.get("topk_blocks", 2048))
        self.block_size = int(config.get("block_size", 8))
        self.row_repeat = int(config.get("row_repeat", 1))
        if self.dtype != torch.float32:
            raise ValueError("select_candidate_blocks only supports dtype=float32")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32/int32")
        state.add_summary(
            "Shape",
            f"({self.num_rows}x{self.width} topk={self.topk_blocks} "
            f"block={self.block_size})",
        )
        logits_elements = self.num_rows * self.width
        output_elements = self.num_rows * self.topk_blocks
        bound_rows = (self.num_rows + self.row_repeat - 1) // self.row_repeat
        state.add_element_count(logits_elements)
        state.add_global_memory_reads((logits_elements + bound_rows) * 4)
        state.add_global_memory_writes(output_elements * 4)

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
        out = torch.empty(
            (num_rows, topk_blocks), device=device, dtype=torch.int32
        )
        return logits, ends, out

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            logits, ends, out = self._prepare(dev_id)
        return self.make_launcher(
            dev_id,
            torch.ops._C.select_candidate_blocks,
            logits,
            None,
            ends,
            self.topk_blocks,
            self.block_size,
            out,
            self.row_repeat,
        )

    def run_verification(self, dev_id):
        verify_rows = 4
        verify_width = 4097
        verify_topk = min(self.topk_blocks, 257)
        logits, ends, actual = self._prepare(
            dev_id, verify_rows, verify_width, verify_topk
        )
        ends[0] = 0
        ends[1] = 2049
        ends[2] = verify_width - 3
        torch.ops._C.select_candidate_blocks(
            logits,
            None,
            ends,
            verify_topk,
            self.block_size,
            actual,
            self.row_repeat,
        )
        scores = _block_scores(
            logits, None, ends, self.block_size, self.row_repeat
        )
        torch.cuda.synchronize(dev_id)
        passed = _matches_topk(actual, scores)
        return bool(passed), 0.0 if passed else 1.0
