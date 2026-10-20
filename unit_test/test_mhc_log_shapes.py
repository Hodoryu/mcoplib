"""Compare the decomposed SGLang MHC path with the native mcoplib path.

The token counts below come from
``266024_Server-Metax-TP8-MHCShape-sglang.log.txt``.  Run on C500 with::

    pytest -q -s unit_test/test_mhc_log_shapes.py

The baseline pre path deliberately calls the public helpers in
``mcoplib.triton_sglang_mhc``.  Those helpers launch, in order,
``_hc_mix_stats_partial_kernel``, ``_hc_mix_stats_reduce_kernel``,
``hc_split_sinkhorn_kernel`` and ``_hc_combine_kernel``.  The post baseline
calls that module's TileLang ``mhc_post`` implementation.

Environment variables:

* ``MHC_LOG_SHAPES``: comma-separated subset of token counts (default: all)
* ``MHC_LOG_WARMUP`` / ``MHC_LOG_BENCH_ROUNDS``
* ``MHC_LOG_DECODE_REPEAT`` / ``MHC_LOG_PREFILL_REPEAT``
* ``MHC_LOG_POST_DECODE_REPEAT`` / ``MHC_LOG_POST_PREFILL_REPEAT``
"""

from __future__ import annotations

import os
import statistics

import pytest
import torch


pytest.importorskip("triton")
pytest.importorskip("tilelang")
pytest.importorskip("deep_gemm")

from mcoplib.mhc import mhc_post, mhc_pre
from mcoplib import triton_sglang_mhc as triton_mhc


HC = 4
HIDDEN = 5120
HC_HIDDEN = HC * HIDDEN
MIX = HC * (2 + HC)

RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_REPEAT = 20

# Unique MHC token counts observed in the supplied TP8 log.
LOG_NUM_TOKENS = (
    1,
    4,
    8,
    193,
    2048,
    4096,
    6125,
    12249,
    12255,
    14301,
    16345,
)

_BENCHMARK_RESULTS = []


def _print_table(title: str, headers, rows) -> None:
    """Print a dependency-free ASCII table that aligns in C500 terminals."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(f"\n{title}")
    print(separator)
    print(
        "| "
        + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(headers)
        )
        + " |"
    )
    print(separator)
    for row in rows:
        print(
            "| "
            + " | ".join(
                str(value).rjust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )
    print(separator)


def _print_benchmark_tables() -> None:
    if not _BENCHMARK_RESULTS:
        return

    results = sorted(_BENCHMARK_RESULTS, key=lambda item: item["tokens"])
    pre_rows = []
    post_rows = []
    for result in results:
        pre_rows.append(
            (
                result["tokens"],
                result["phase"].lower(),
                f'{result["triton_pre_ms"]:.4f}',
                f'{result["native_pre_ms"]:.4f}',
                f'{result["pre_speedup"]:.2f}x',
                f'{result["post_mix_error"]:.3g}',
                f'{result["comb_mix_error"]:.3g}',
                f'{result["layer_input_error"]:.3g}',
            )
        )
        post_rows.append(
            (
                result["tokens"],
                result["phase"].lower(),
                f'{result["triton_post_ms"]:.4f}',
                f'{result["native_post_ms"]:.4f}',
                f'{result["post_speedup"]:.2f}x',
                f'{result["post_output_error"]:.3g}',
            )
        )

    _print_table(
        "MHC pre: decomposed Triton path vs mhc_pre",
        (
            "N",
            "phase",
            "Triton(ms)",
            "mhc_pre(ms)",
            "speedup",
            "post_err",
            "comb_err",
            "layer_err",
        ),
        pre_rows,
    )
    _print_table(
        "MHC post: TileLang mhc_post vs mcoplib.mhc.mhc_post",
        (
            "N",
            "phase",
            "TileLang(ms)",
            "mhc_post(ms)",
            "speedup",
            "max_err",
        ),
        post_rows,
    )


@pytest.fixture(scope="module", autouse=True)
def _benchmark_summary():
    """Print one comparison table after all selected log shapes finish."""
    _BENCHMARK_RESULTS.clear()
    yield
    _print_benchmark_tables()


@pytest.fixture(autouse=True)
def _standalone_sglang_context(monkeypatch):
    """Keep the legacy post wrapper independent of SGLang server state.

    ``triton_sglang_mhc.mhc_post`` queries the process-global SGLang parallel
    context only to decide whether its inputs need ``strict_contiguous``.
    Operator unit tests do not publish that server configuration, and every
    post input created below is already contiguous.  Pinning the predicate to
    false therefore preserves the normal local-kernel path without requiring
    a fake distributed SGLang runtime.
    """
    monkeypatch.setattr(
        triton_mhc,
        "is_dsa_prefill_cp_round_robin_split",
        lambda: False,
    )


def _selected_shapes() -> tuple[int, ...]:
    value = os.getenv("MHC_LOG_SHAPES")
    if value is None:
        return LOG_NUM_TOKENS
    selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = set(selected) - set(LOG_NUM_TOKENS)
    if unknown:
        raise ValueError(
            f"MHC_LOG_SHAPES contains values not present in the log: {sorted(unknown)}"
        )
    return selected


def _make_inputs(num_tokens: int):
    # Scaling fn keeps logits in a useful range while preserving a realistic
    # BF16-residual/FP32-weight workload.
    torch.manual_seed(266024 + num_tokens)
    residual = torch.randn(
        (num_tokens, HC, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn((MIX, HC_HIDDEN), device="cuda", dtype=torch.float32)
    fn.mul_(HC_HIDDEN**-0.5)
    hc_scale = torch.tensor((0.8, 1.1, 0.9), device="cuda", dtype=torch.float32)
    hc_base = torch.randn((MIX,), device="cuda", dtype=torch.float32) * 0.05
    return residual, fn, hc_scale, hc_base


def _run_triton_pre(residual, fn, hc_scale, hc_base):
    """Launch the four decomposed baseline stages in their production order."""
    num_tokens = residual.shape[0]
    x_flat = residual.view(num_tokens, HC_HIDDEN)

    # hc_mix_stats launches _hc_mix_stats_partial_kernel followed by
    # _hc_mix_stats_reduce_kernel.
    mixes = triton_mhc.hc_mix_stats(x_flat, fn, RMS_EPS)

    # On C500 this wrapper launches hc_split_sinkhorn_kernel.
    pre, post, comb = triton_mhc.hc_split_sinkhorn(
        mixes.view(num_tokens, 1, MIX),
        hc_scale,
        hc_base,
        hc_mult=HC,
        sinkhorn_iters=SINKHORN_REPEAT,
        eps=HC_SINKHORN_EPS,
    )
    pre = pre.view(num_tokens, HC)
    post = post.view(num_tokens, HC, 1)
    comb = comb.view(num_tokens, HC, HC)

    # hc_combine launches _hc_combine_kernel.
    layer_input = triton_mhc.hc_combine(
        x_flat, pre, HC, out_dtype=torch.bfloat16
    )
    return post, comb, layer_input


def _run_native_pre(residual, fn, hc_scale, hc_base):
    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )


def _benchmark_ms(fn, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _benchmark_pair_ms(lhs, rhs, warmup: int, repeat: int, rounds: int):
    """Alternate measurement order and return each side's median latency."""
    for _ in range(warmup):
        lhs()
        rhs()
    torch.cuda.synchronize()

    lhs_times = []
    rhs_times = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            lhs_times.append(_benchmark_ms(lhs, repeat))
            rhs_times.append(_benchmark_ms(rhs, repeat))
        else:
            rhs_times.append(_benchmark_ms(rhs, repeat))
            lhs_times.append(_benchmark_ms(lhs, repeat))
    return statistics.median(lhs_times), statistics.median(rhs_times)


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/MACA GPU")
@pytest.mark.parametrize("num_tokens", _selected_shapes())
def test_mhc_pre_post_on_log_shape(num_tokens: int):
    residual, fn, hc_scale, hc_base = _make_inputs(num_tokens)

    # First calls also force lazy loading/JIT; they are not timed.
    triton_post_mix, triton_comb_mix, triton_layer_input = _run_triton_pre(
        residual, fn, hc_scale, hc_base
    )
    native_post_mix, native_comb_mix, native_layer_input = _run_native_pre(
        residual, fn, hc_scale, hc_base
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        native_post_mix, triton_post_mix, rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        native_comb_mix, triton_comb_mix, rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        native_layer_input, triton_layer_input, rtol=2e-2, atol=2e-2
    )

    # Feed both post implementations exactly the same values.  Reusing the
    # pre output also checks the [N, 4, 1] -> [N, 4] accepted layout boundary.
    layer_output = torch.randn(
        (num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16
    )

    def run_triton_post():
        return triton_mhc.mhc_post(
            layer_output, residual, triton_post_mix, triton_comb_mix
        )

    def run_native_post():
        return mhc_post(
            layer_output,
            residual,
            triton_post_mix,
            triton_comb_mix,
        )

    triton_post_output = run_triton_post()
    native_post_output = run_native_post()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        native_post_output, triton_post_output, rtol=2e-2, atol=2e-2
    )

    warmup = int(os.getenv("MHC_LOG_WARMUP", "3"))
    rounds = int(os.getenv("MHC_LOG_BENCH_ROUNDS", "3"))
    phase = "DECODE" if num_tokens <= 193 else "PREFILL"
    pre_default = "20" if phase == "DECODE" else "3"
    post_default = "50" if phase == "DECODE" else "10"
    pre_repeat = int(os.getenv(f"MHC_LOG_{phase}_REPEAT", pre_default))
    post_repeat = int(
        os.getenv(f"MHC_LOG_POST_{phase}_REPEAT", post_default)
    )

    triton_pre_ms, native_pre_ms = _benchmark_pair_ms(
        lambda: _run_triton_pre(residual, fn, hc_scale, hc_base),
        lambda: _run_native_pre(residual, fn, hc_scale, hc_base),
        warmup=warmup,
        repeat=pre_repeat,
        rounds=rounds,
    )
    triton_post_ms, native_post_ms = _benchmark_pair_ms(
        run_triton_post,
        run_native_post,
        warmup=warmup,
        repeat=post_repeat,
        rounds=rounds,
    )

    pre_errors = (
        _max_abs(native_post_mix, triton_post_mix),
        _max_abs(native_comb_mix, triton_comb_mix),
        _max_abs(native_layer_input, triton_layer_input),
    )
    post_error = _max_abs(native_post_output, triton_post_output)
    _BENCHMARK_RESULTS.append(
        {
            "tokens": num_tokens,
            "phase": phase,
            "triton_pre_ms": triton_pre_ms,
            "native_pre_ms": native_pre_ms,
            "pre_speedup": triton_pre_ms / native_pre_ms,
            "post_mix_error": pre_errors[0],
            "comb_mix_error": pre_errors[1],
            "layer_input_error": pre_errors[2],
            "triton_post_ms": triton_post_ms,
            "native_post_ms": native_post_ms,
            "post_speedup": triton_post_ms / native_post_ms,
            "post_output_error": post_error,
        }
    )

