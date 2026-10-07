"""Benchmark the direct SV-010 wrapper against the current product semantic."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import torch

BENCHMARK = "statevector_local_diagonal"
RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_local_diagonal.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_local_diagonal.v1"
SEMANTIC_ID = "statevector.apply.diagonal.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-010-A"
RUNNER = "benchmarks/internal/evidence/statevector_local_diagonal_probe.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 16, (0,), False),
    (1, 20, (19,), False),
    (1, 20, (0, 19), False),
    (1, 24, (3, 19), False),
    (4, 20, (1, 18), True),
)


class TimingResult(TypedDict):
    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float


def _measure_group(operation: Callable[[], Any], group_size: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(group_size):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / group_size


def _measure_pair(
    direct: Callable[[], Any],
    reference: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> tuple[TimingResult, TimingResult]:
    for index in range(warmup):
        if index % 2:
            reference()
            direct()
        else:
            direct()
            reference()
    torch.cuda.synchronize()

    direct_samples = []
    reference_samples = []
    for index in range(repeats):
        if index % 2:
            reference_samples.append(_measure_group(reference, group_size))
            direct_samples.append(_measure_group(direct, group_size))
        else:
            direct_samples.append(_measure_group(direct, group_size))
            reference_samples.append(_measure_group(reference, group_size))
    return (
        {
            "samples_seconds_per_invocation": direct_samples,
            "median_seconds_per_invocation": statistics.median(direct_samples),
        },
        {
            "samples_seconds_per_invocation": reference_samples,
            "median_seconds_per_invocation": statistics.median(reference_samples),
        },
    )


def _shape_record(
    batch: int,
    n_qubits: int,
    qubits: tuple[int, ...],
    batched_diagonal: bool,
) -> dict[str, object]:
    return {
        "batch": batch,
        "n_qubits": n_qubits,
        "qubits": list(qubits),
        "batched_diagonal": batched_diagonal,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed SV-010 matrix."""

    from flagquantum.kernels.provenance import triton_compiler_provenance
    from flagquantum.kernels.triton.statevector_diagonal import (
        apply_complex64_local_diagonal,
    )
    from flagquantum.simulation.statevector.operations import _apply_diagonal_matrix

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-010 benchmark evidence")

    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected_distribution = (
        "triton" if args.compiler_lane == "stock_triton" else "flagtree"
    )
    if distribution != expected_distribution or identity_status != "resolved":
        raise RuntimeError(
            f"lane {args.compiler_lane!r} requires {expected_distribution!r}, "
            f"found {distribution!r} with identity {identity_status!r}"
        )

    cases = []
    for index, (batch, n_qubits, qubits, batched_diagonal) in enumerate(SHAPE_MATRIX):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            1 << n_qubits,
            generator=generator,
            dtype=torch.complex64,
            device="cuda",
        )
        diagonal_shape = (
            (batch, 1 << len(qubits)) if batched_diagonal else (1 << len(qubits),)
        )
        diagonal = torch.randn(
            *diagonal_shape,
            generator=generator,
            dtype=torch.complex64,
            device="cuda",
        )
        matrix = torch.diag_embed(diagonal)

        direct = partial(
            apply_complex64_local_diagonal,
            state,
            diagonal,
            qubits=qubits,
        )
        reference = partial(_apply_diagonal_matrix, state, matrix, qubits, n_qubits)

        expected = reference()
        actual = direct()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        direct_timing, reference_timing = _measure_pair(
            direct,
            reference,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        difference = actual - expected
        cases.append(
            {
                "shape": _shape_record(batch, n_qubits, qubits, batched_diagonal),
                "direct_kernel_wrapper": direct_timing,
                "pytorch_product_reference": reference_timing,
                "speedup_over_pytorch": reference_timing[
                    "median_seconds_per_invocation"
                ]
                / direct_timing["median_seconds_per_invocation"],
                "maximum_absolute_error": float(torch.max(torch.abs(difference))),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected).clamp_min(
                        torch.finfo(torch.float32).eps
                    )
                ),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": args.source_revision,
        "command": shlex.join(sys.argv),
        "host_label": args.host_label,
        "reported_hostname": socket.gethostname(),
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": args.compiler_lane,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "integration_path": integration_path,
            "identity_status": identity_status,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": args.group_size,
            "ordering": "counterbalanced by repeat parity",
            "synchronization": "before and after every timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "seed": args.seed,
        "cases": cases,
    }


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-010 evidence requires exactly four raw runs")
    if any(run.get("schema") != RUN_SCHEMA for run in runs):
        raise ValueError("SV-010 raw-run schema mismatch")
    identities = {(run["host_label"], run["compiler_lane"]) for run in runs}
    expected_identities = {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    if identities != expected_identities:
        raise ValueError("SV-010 evidence host/compiler matrix is incomplete")
    revisions = {run["source_revision"] for run in runs}
    measurements = {json.dumps(run["measurement"], sort_keys=True) for run in runs}
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("SV-010 evidence must use one source and measurement policy")
    expected_shapes = [_shape_record(*shape) for shape in SHAPE_MATRIX]
    if any([case["shape"] for case in run["cases"]] != expected_shapes for run in runs):
        raise ValueError("SV-010 evidence shape matrix mismatch")

    cases = [case for run in runs for case in run["cases"]]
    speedups = [float(case["speedup_over_pytorch"]) for case in cases]
    maximum_absolute_error = max(
        float(case["maximum_absolute_error"]) for case in cases
    )
    maximum_relative_l2_error = max(float(case["relative_l2_error"]) for case in cases)
    all_cases_win = all(speedup > 1.0 for speedup in speedups)
    return {
        "benchmark": BENCHMARK,
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": revisions.pop(),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-device kernel benchmark is not distributed scalability evidence"
        ],
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "measurement": runs[0]["measurement"],
        "shape_matrix": expected_shapes,
        "runs": sorted(runs, key=lambda run: (run["host_label"], run["compiler_lane"])),
        "aggregate": {
            "case_count": len(cases),
            "minimum_speedup_over_pytorch": min(speedups),
            "maximum_speedup_over_pytorch": max(speedups),
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_l2_error,
            "all_cases_win": all_cases_win,
            "decision": (
                "eligible_for_dispatch_evaluation"
                if all_cases_win
                else "retain_experimental"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--host-label", choices=HOSTS, required=True)
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261_006)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        collect_run(args) if args.command_name == "run" else aggregate_runs(args.input)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
