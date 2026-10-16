"""Collect two-host public-dispatch evidence for SV-010."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

BENCHMARK = "statevector_local_diagonal_dispatch"
RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_local_diagonal_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_local_diagonal_dispatch.v1"
SEMANTIC_ID = "statevector.apply.diagonal.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-010-A"
RUNNER = "benchmarks/statevector_local_diagonal_dispatch.py"
DISPATCH_VARIABLE = "FQ_TRITON_DIAGONAL_MATRIX"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 16, (0,), False),
    (1, 20, (19,), False),
    (1, 20, (0, 19), False),
    (1, 24, (3, 19), False),
    (4, 20, (1, 18), True),
)
PERFORMANCE_FLOOR = 1.0
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAXIMUM_ABSOLUTE_ERROR = 2.0e-5
_MAXIMUM_RELATIVE_L2_ERROR = 1.0e-6


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
    dispatch: Callable[[], Any],
    reference: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> tuple[TimingResult, TimingResult]:
    for index in range(warmup):
        if index % 2:
            reference()
            dispatch()
        else:
            dispatch()
            reference()
    torch.cuda.synchronize()
    dispatch_samples = []
    reference_samples = []
    for index in range(repeats):
        if index % 2:
            reference_samples.append(_measure_group(reference, group_size))
            dispatch_samples.append(_measure_group(dispatch, group_size))
        else:
            dispatch_samples.append(_measure_group(dispatch, group_size))
            reference_samples.append(_measure_group(reference, group_size))
    return (
        {
            "samples_seconds_per_invocation": dispatch_samples,
            "median_seconds_per_invocation": statistics.median(dispatch_samples),
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
    """Execute one host/compiler lane of the fixed public dispatch matrix."""

    from flagquantum.simulation.statevector.diagonal_matrix_dispatch import (
        _diagonal_matrix_kernel_enabled,
    )
    from flagquantum.simulation.statevector.operations import (
        _apply_diagonal_matrix,
        _apply_diagonal_matrix_reference,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-010 dispatch evidence")
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

    os.environ[DISPATCH_VARIABLE] = "1"
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
        matrix = torch.diag_embed(diagonal).contiguous()
        if not _diagonal_matrix_kernel_enabled(
            state,
            matrix,
            qubits=qubits,
            n_qubits=n_qubits,
        ):
            raise RuntimeError("the fixed SV-010 case did not select the kernel")

        dispatch = partial(_apply_diagonal_matrix, state, matrix, qubits, n_qubits)
        reference = partial(
            _apply_diagonal_matrix_reference,
            state,
            matrix,
            qubits,
            n_qubits,
        )
        actual = dispatch()
        expected = reference()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        dispatch_timing, reference_timing = _measure_pair(
            dispatch,
            reference,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        difference = actual - expected
        cases.append(
            {
                "shape": _shape_record(batch, n_qubits, qubits, batched_diagonal),
                "public_catalog_dispatch": dispatch_timing,
                "public_pytorch_reference": reference_timing,
                "public_speedup_over_pytorch": reference_timing[
                    "median_seconds_per_invocation"
                ]
                / dispatch_timing["median_seconds_per_invocation"],
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
        "execution_semantics": "single_device_public_runtime_dispatch",
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
        "performance_floor": PERFORMANCE_FLOOR,
        "seed": args.seed,
        "cases": cases,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def validate_run(payload: Mapping[str, Any]) -> None:
    """Fail closed when one raw public-dispatch run misses its contract."""

    if payload.get("schema") != RUN_SCHEMA:
        raise ValueError("SV-010 dispatch raw-run schema mismatch")
    if payload.get("semantic_id") != SEMANTIC_ID:
        raise ValueError("SV-010 dispatch semantic mismatch")
    if payload.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("SV-010 dispatch implementation mismatch")
    if payload.get("runner") != RUNNER:
        raise ValueError("SV-010 dispatch runner mismatch")
    if _FULL_REVISION.fullmatch(str(payload.get("source_revision"))) is None:
        raise ValueError("SV-010 dispatch source revision must be a full Git hash")
    if payload.get("host_label") not in HOSTS:
        raise ValueError("SV-010 dispatch host is invalid")
    if payload.get("compiler_lane") not in COMPILER_LANES:
        raise ValueError("SV-010 dispatch compiler lane is invalid")
    cases = payload.get("cases")
    if not isinstance(cases, list) or [case.get("shape") for case in cases] != [
        _shape_record(*shape) for shape in SHAPE_MATRIX
    ]:
        raise ValueError("SV-010 dispatch shape matrix mismatch")
    repeats = _mapping(payload.get("measurement"), "measurement").get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SV-010 dispatch repeats must be positive")
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        medians = {}
        for name in ("public_catalog_dispatch", "public_pytorch_reference"):
            result = _mapping(case.get(name), name)
            samples = result.get("samples_seconds_per_invocation")
            if not isinstance(samples, list) or len(samples) != repeats:
                raise ValueError(f"SV-010 dispatch case {index} samples are invalid")
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"SV-010 dispatch case {index} median is invalid")
            medians[name] = median
        speedup = (
            medians["public_pytorch_reference"] / medians["public_catalog_dispatch"]
        )
        if case.get("public_speedup_over_pytorch") != speedup:
            raise ValueError(f"SV-010 dispatch case {index} speedup is invalid")
        if speedup < PERFORMANCE_FLOOR:
            raise ValueError(f"SV-010 dispatch case {index} misses performance floor")
        if (
            float(case.get("maximum_absolute_error", float("inf")))
            > _MAXIMUM_ABSOLUTE_ERROR
        ):
            raise ValueError(f"SV-010 dispatch case {index} exceeds absolute error")
        if (
            float(case.get("relative_l2_error", float("inf")))
            > _MAXIMUM_RELATIVE_L2_ERROR
        ):
            raise ValueError(f"SV-010 dispatch case {index} exceeds relative error")


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-010 dispatch evidence requires exactly four raw runs")
    for run in runs:
        validate_run(run)
    identities = {(run["host_label"], run["compiler_lane"]) for run in runs}
    if identities != {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }:
        raise ValueError("SV-010 dispatch host/compiler matrix is incomplete")
    revisions = {run["source_revision"] for run in runs}
    measurements = {json.dumps(run["measurement"], sort_keys=True) for run in runs}
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("SV-010 dispatch evidence must share source and policy")
    cases = [case for run in runs for case in run["cases"]]
    speedups = [float(case["public_speedup_over_pytorch"]) for case in cases]
    return {
        "benchmark": BENCHMARK,
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": revisions.pop(),
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "distribution_semantics": "single_device_fast_path",
        "claim_evidence_type": "development_smoke",
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-device dispatch benchmark is not distributed scalability evidence"
        ],
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "measurement": runs[0]["measurement"],
        "shape_matrix": [_shape_record(*shape) for shape in SHAPE_MATRIX],
        "runs": sorted(runs, key=lambda run: (run["host_label"], run["compiler_lane"])),
        "aggregate": {
            "case_count": len(cases),
            "minimum_public_speedup_over_pytorch": min(speedups),
            "maximum_public_speedup_over_pytorch": max(speedups),
            "maximum_absolute_error": max(
                float(case["maximum_absolute_error"]) for case in cases
            ),
            "maximum_relative_l2_error": max(
                float(case["relative_l2_error"]) for case in cases
            ),
            "all_cases_meet_performance_floor": all(
                speedup >= PERFORMANCE_FLOOR for speedup in speedups
            ),
            "decision": "enable_default_dispatch",
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
