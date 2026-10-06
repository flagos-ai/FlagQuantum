"""Probe a controlled one-qubit kernel against the layout/BMM reference."""

from __future__ import annotations

import argparse
import json
import platform
import re
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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_controlled_matrix.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_controlled_matrix.v1"
SEMANTIC_ID = "statevector.apply.controlled_matrix_1q.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-012-A"
RUNNER = "benchmarks/internal/evidence/statevector_controlled_matrix_probe.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 1 << 16, 0, 1, False),
    (1, 1 << 20, 0, 19, False),
    (1, 1 << 20, 17, 3, False),
    (1, 1 << 24, 3, 19, False),
    (4, 1 << 20, 1, 18, True),
)
RESULT_NAMES = ("direct_kernel_wrapper", "pytorch_layout_bmm_reference")
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAXIMUM_ABSOLUTE_ERROR = 2.0e-5
_MAXIMUM_RELATIVE_L2_ERROR = 1.0e-6


class TimingResult(TypedDict):
    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float


def _reference(
    state: torch.Tensor,
    controlled_matrix: torch.Tensor,
    control_qubit: int,
    target_qubit: int,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    return _apply_matrix_layout(
        state,
        controlled_matrix,
        (control_qubit, target_qubit),
        state.shape[1].bit_length() - 1,
    )


def _measure_group(operation: Callable[[], Any], group_size: int) -> float:
    started = time.perf_counter()
    for _ in range(group_size):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / group_size


def _measure_pair(
    first: Callable[[], Any],
    second: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> tuple[TimingResult, TimingResult]:
    for _ in range(warmup):
        first()
        second()
    torch.cuda.synchronize()
    first_samples = []
    second_samples = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            first_samples.append(_measure_group(first, group_size))
            second_samples.append(_measure_group(second, group_size))
        else:
            second_samples.append(_measure_group(second, group_size))
            first_samples.append(_measure_group(first, group_size))
    return (
        {
            "samples_seconds_per_invocation": first_samples,
            "median_seconds_per_invocation": statistics.median(first_samples),
        },
        {
            "samples_seconds_per_invocation": second_samples,
            "median_seconds_per_invocation": statistics.median(second_samples),
        },
    )


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed SV-012 matrix."""

    from flagquantum.kernels.provenance import triton_compiler_provenance
    from flagquantum.kernels.triton.statevector_controlled_matrix import (
        apply_complex64_local_controlled_1q,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-012 benchmark evidence")
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
    for index, (
        batch,
        amplitudes,
        control_qubit,
        target_qubit,
        batched_matrix,
    ) in enumerate(SHAPE_MATRIX):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            amplitudes,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        matrix_shape = (batch, 2, 2) if batched_matrix else (2, 2)
        matrix = torch.randn(
            *matrix_shape,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        matrices = matrix.reshape(-1, 2, 2)
        if matrices.shape[0] == 1:
            matrices = matrices.expand(batch, -1, -1)
        controlled = torch.zeros(
            batch,
            4,
            4,
            device="cuda",
            dtype=torch.complex64,
        )
        controlled[:, :2, :2] = torch.eye(
            2,
            device="cuda",
            dtype=torch.complex64,
        )
        controlled[:, 2:, 2:] = matrices
        expected = _reference(
            state,
            controlled,
            control_qubit,
            target_qubit,
        )
        actual = apply_complex64_local_controlled_1q(
            state,
            matrix,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        direct, reference = _measure_pair(
            partial(
                apply_complex64_local_controlled_1q,
                state,
                matrix,
                control_qubit=control_qubit,
                target_qubit=target_qubit,
            ),
            partial(
                _reference,
                state,
                controlled,
                control_qubit,
                target_qubit,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        difference = actual - expected
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "amplitudes_per_batch": amplitudes,
                    "control_qubit": control_qubit,
                    "target_qubit": target_qubit,
                    "batched_matrix": batched_matrix,
                },
                "direct_kernel_wrapper": direct,
                "pytorch_layout_bmm_reference": reference,
                "speedup_over_pytorch": reference["median_seconds_per_invocation"]
                / direct["median_seconds_per_invocation"],
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
            "ordering": "alternating direct-first and reference-first groups",
            "synchronization": "once after warmup and once per timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "seed": args.seed,
        "cases": cases,
    }


def _validate_run(run: dict[str, Any]) -> None:
    expected_fields = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }
    for field, expected in expected_fields.items():
        if run.get(field) != expected:
            raise ValueError(f"SV-012 run field {field!r} is invalid")
    if _FULL_REVISION.fullmatch(str(run.get("source_revision"))) is None:
        raise ValueError("SV-012 source revision must be a full Git hash")
    host = run.get("host_label")
    lane = run.get("compiler_lane")
    if host not in HOSTS or lane not in COMPILER_LANES:
        raise ValueError("SV-012 run host or compiler lane is invalid")
    compiler = run.get("compiler")
    if not isinstance(compiler, dict):
        raise ValueError("SV-012 compiler identity is missing")
    expected_compiler = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[lane]
    if (
        compiler.get("distribution"),
        compiler.get("integration_path"),
    ) != expected_compiler or compiler.get("identity_status") != "resolved":
        raise ValueError("SV-012 compiler identity does not match its lane")
    measurement = run.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("SV-012 measurement policy is missing")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SV-012 repeats must be positive")
    expected_shapes = [
        {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "control_qubit": control_qubit,
            "target_qubit": target_qubit,
            "batched_matrix": batched_matrix,
        }
        for batch, amplitudes, control_qubit, target_qubit, batched_matrix in SHAPE_MATRIX
    ]
    cases = run.get("cases")
    if (
        not isinstance(cases, list)
        or [case.get("shape") for case in cases] != expected_shapes
    ):
        raise ValueError("SV-012 evidence shape matrix mismatch")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"SV-012 case {index} must be an object")
        medians = {}
        for result_name in RESULT_NAMES:
            result = case.get(result_name)
            if not isinstance(result, dict):
                raise ValueError(f"SV-012 case {index} result is missing")
            samples = result.get("samples_seconds_per_invocation")
            if (
                not isinstance(samples, list)
                or len(samples) != repeats
                or any(
                    not isinstance(value, (int, float)) or value <= 0
                    for value in samples
                )
            ):
                raise ValueError(f"SV-012 case {index} samples are invalid")
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"SV-012 case {index} median is invalid")
            medians[result_name] = median
        speedup = (
            medians["pytorch_layout_bmm_reference"] / medians["direct_kernel_wrapper"]
        )
        if case.get("speedup_over_pytorch") != speedup:
            raise ValueError(f"SV-012 case {index} speedup is invalid")
        absolute_error = case.get("maximum_absolute_error")
        relative_error = case.get("relative_l2_error")
        if (
            not isinstance(absolute_error, (int, float))
            or absolute_error > _MAXIMUM_ABSOLUTE_ERROR
            or not isinstance(relative_error, (int, float))
            or relative_error > _MAXIMUM_RELATIVE_L2_ERROR
        ):
            raise ValueError(f"SV-012 case {index} exceeds its error tolerance")


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-012 evidence requires exactly four raw runs")
    for run in runs:
        _validate_run(run)
    identities = {(run["host_label"], run["compiler_lane"]) for run in runs}
    expected_identities = {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    if identities != expected_identities:
        raise ValueError("SV-012 evidence host/compiler matrix is incomplete")
    revisions = {run["source_revision"] for run in runs}
    measurements = {json.dumps(run["measurement"], sort_keys=True) for run in runs}
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("SV-012 evidence must use one source and measurement policy")
    expected_shapes = [
        {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "control_qubit": control_qubit,
            "target_qubit": target_qubit,
            "batched_matrix": batched_matrix,
        }
        for batch, amplitudes, control_qubit, target_qubit, batched_matrix in SHAPE_MATRIX
    ]
    if any([case["shape"] for case in run["cases"]] != expected_shapes for run in runs):
        raise ValueError("SV-012 evidence shape matrix mismatch")

    cases = [case for run in runs for case in run["cases"]]
    speedups = [float(case["speedup_over_pytorch"]) for case in cases]
    maximum_absolute_error = max(
        float(case["maximum_absolute_error"]) for case in cases
    )
    maximum_relative_l2_error = max(float(case["relative_l2_error"]) for case in cases)
    all_cases_win = all(speedup > 1.0 for speedup in speedups)
    return {
        "benchmark": "statevector_controlled_matrix",
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
    run.add_argument("--seed", type=int, default=261_012)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        collect_run(args) if args.command_name == "run" else aggregate_runs(args.input)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
