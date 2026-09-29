"""Measure the native dual-state CPU gather against two PyTorch gathers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from flagquantum.simulation.native_cpu import fused_cx_adjoint_gather
from flagquantum.simulation.statevector.operations import (
    _cx_sequence_permutation_index,
)


def _measurement(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    deviations = [abs(value - median) for value in samples]
    return {
        "median_seconds": median,
        "samples_seconds": samples,
        "relative_median_absolute_deviation": (
            statistics.median(deviations) / median if median else 0.0
        ),
    }


def _time_call(call: Callable[[], object], *, calls_per_sample: int) -> float:
    start = time.perf_counter()
    result = None
    for _ in range(calls_per_sample):
        result = call()
    elapsed = (time.perf_counter() - start) / calls_per_sample
    if result is None:
        raise RuntimeError("native CPU CX adjoint gather is unavailable")
    return elapsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    width = 1 << args.n_wires
    real = torch.randn((1, width), dtype=torch.float64)
    imag = torch.randn((1, width), dtype=torch.float64)
    ket = torch.complex(real, imag)
    real = torch.randn((1, width), dtype=torch.float64)
    imag = torch.randn((1, width), dtype=torch.float64)
    adjoint = torch.complex(real, imag)
    controls = tuple(range(args.n_wires - 1))
    targets = tuple(range(1, args.n_wires))
    index = _cx_sequence_permutation_index(
        controls,
        targets,
        args.n_wires,
        device=ket.device,
        dtype=ket.dtype,
    )

    expected = (
        torch.index_select(ket, 1, index),
        torch.index_select(adjoint, 1, index),
    )
    actual = fused_cx_adjoint_gather(ket, adjoint, index)
    if actual is None:
        raise RuntimeError(
            "build the native CPU extension before running this benchmark"
        )
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    rows: list[dict[str, Any]] = []
    for threads in args.threads:
        torch.set_num_threads(threads)

        def baseline() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.index_select(ket, 1, index),
                torch.index_select(adjoint, 1, index),
            )

        def native() -> tuple[torch.Tensor, torch.Tensor] | None:
            return fused_cx_adjoint_gather(ket, adjoint, index)

        for _ in range(args.warmup):
            baseline()
            native()
        samples: dict[str, list[float]] = {"baseline": [], "native": []}
        calls = {"baseline": baseline, "native": native}
        for iteration in range(args.iterations):
            order = (
                ("baseline", "native") if iteration % 2 == 0 else ("native", "baseline")
            )
            for name in order:
                samples[name].append(
                    _time_call(calls[name], calls_per_sample=args.calls_per_sample)
                )
        baseline_result = _measurement(samples["baseline"])
        native_result = _measurement(samples["native"])
        rows.append(
            {
                "threads": threads,
                "baseline": baseline_result,
                "native": native_result,
                "speedup": baseline_result["median_seconds"]
                / native_result["median_seconds"],
            }
        )

    return {
        "schema": "flagquantum.native_cpu_cx_adjoint_gather.v1",
        "benchmark": "native_cpu_cx_adjoint_gather",
        "artifact_class": "measured_microbenchmark_non_release",
        "benchmark_evidence_class": "comparison_non_release",
        "non_release_evidence": True,
        "distribution_semantics": "single_device_fast_path",
        "workload": {
            "n_wires": args.n_wires,
            "batch_size": 1,
            "dtype": "complex128",
            "state_count": 2,
            "cx_count": len(controls),
            "logical_state_bytes_each": ket.numel() * ket.element_size(),
        },
        "methodology": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "calls_per_sample": args.calls_per_sample,
            "alternating_order": True,
            "baseline": "two torch.index_select calls",
            "native": "one fused dual-state C++ gather",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        },
        "correctness": {"passed": True, "absolute_tolerance": 0.0},
        "results": rows,
        "world_size": 1,
        "scalability_claim_allowed": False,
        "scalability_blockers": ["single_device_microbenchmark"],
        "release_gate_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=22)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
