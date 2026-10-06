"""Collect two-host public-dispatch evidence for SV-009."""

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
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_local_2q_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_local_2q_dispatch.v1"
SEMANTIC_ID = "statevector.apply.matrix_2q.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-009-A"
RUNNER = "benchmarks/statevector_local_2q_dispatch.py"
DISPATCH_VARIABLE = "FQ_TRITON_TWO_QUBIT_MATRIX"
RESULT_NAMES = ("public_catalog_dispatch", "public_pytorch_reference")
PERFORMANCE_FLOOR = 1.0
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 1 << 16, 0, 1),
    (1, 1 << 20, 0, 19),
    (1, 1 << 24, 23, 22),
    (1, 1 << 24, 23, 0),
    (4, 1 << 20, 10, 3),
)
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAXIMUM_ABSOLUTE_ERROR = 2.0e-5
_MAXIMUM_RELATIVE_L2_ERROR = 1.0e-6


class TimingResult(TypedDict):
    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float


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


def _public_dispatch(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: tuple[int, int],
    n_wires: int,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix

    return _apply_matrix(state, matrix, wires, n_wires)


def _public_reference(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: tuple[int, int],
    n_wires: int,
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    return _apply_matrix_layout(state, matrix, wires, n_wires)


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed public dispatch matrix."""

    from flagquantum.simulation.statevector.two_qubit_matrix_dispatch import (
        _two_qubit_matrix_kernel_enabled,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-009 dispatch evidence")
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
    for index, (
        batch,
        amplitudes,
        first_bit_position,
        second_bit_position,
    ) in enumerate(SHAPE_MATRIX):
        n_wires = amplitudes.bit_length() - 1
        wires = (
            n_wires - 1 - first_bit_position,
            n_wires - 1 - second_bit_position,
        )
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            amplitudes,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        matrix = torch.randn(
            4,
            4,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        if not _two_qubit_matrix_kernel_enabled(
            state,
            matrix,
            wires=wires,
            n_wires=n_wires,
        ):
            raise RuntimeError("the fixed SV-009 case did not select the kernel")
        actual = _public_dispatch(state, matrix, wires, n_wires)
        expected = _public_reference(state, matrix, wires, n_wires)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        dispatched, reference = _measure_pair(
            partial(_public_dispatch, state, matrix, wires, n_wires),
            partial(_public_reference, state, matrix, wires, n_wires),
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        difference = actual - expected
        speedup = (
            reference["median_seconds_per_invocation"]
            / dispatched["median_seconds_per_invocation"]
        )
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "amplitudes_per_batch": amplitudes,
                    "first_bit_position": first_bit_position,
                    "second_bit_position": second_bit_position,
                },
                "public_catalog_dispatch": dispatched,
                "public_pytorch_reference": reference,
                "public_speedup_over_pytorch": speedup,
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
            "ordering": "alternating dispatch-first and reference-first groups",
            "synchronization": "once after warmup and once per timed group",
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


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def validate_run(payload: Mapping[str, Any]) -> None:
    """Fail closed when one raw public-dispatch run misses its contract."""

    expected_fields = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "performance_floor": PERFORMANCE_FLOOR,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(f"SV-009 dispatch field {field!r} is invalid")
    if _FULL_REVISION.fullmatch(str(payload.get("source_revision"))) is None:
        raise ValueError("SV-009 dispatch source revision must be a full Git hash")
    host = payload.get("host_label")
    lane = payload.get("compiler_lane")
    if host not in HOSTS or lane not in COMPILER_LANES:
        raise ValueError("SV-009 dispatch host or compiler lane is invalid")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected_compiler = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[lane]
    if (
        compiler.get("distribution"),
        compiler.get("integration_path"),
    ) != expected_compiler or compiler.get("identity_status") != "resolved":
        raise ValueError("SV-009 dispatch compiler identity does not match its lane")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SV-009 dispatch repeats must be positive")
    expected_shapes = [
        {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "first_bit_position": first_bit_position,
            "second_bit_position": second_bit_position,
        }
        for batch, amplitudes, first_bit_position, second_bit_position in SHAPE_MATRIX
    ]
    cases = _sequence(payload.get("cases"), "cases")
    if [case.get("shape") for case in cases] != expected_shapes:
        raise ValueError("SV-009 dispatch evidence shape matrix mismatch")
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        medians = {}
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds_per_invocation"),
                f"cases[{index}].{result_name}.samples_seconds_per_invocation",
            )
            if len(samples) != repeats or any(
                not isinstance(value, (int, float)) or value <= 0 for value in samples
            ):
                raise ValueError(f"SV-009 dispatch case {index} samples are invalid")
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"SV-009 dispatch case {index} median is invalid")
            medians[result_name] = median
        speedup = (
            medians["public_pytorch_reference"] / medians["public_catalog_dispatch"]
        )
        if case.get("public_speedup_over_pytorch") != speedup:
            raise ValueError(f"SV-009 dispatch case {index} speedup is invalid")
        if speedup < PERFORMANCE_FLOOR:
            raise ValueError(f"SV-009 dispatch case {index} misses performance floor")
        absolute_error = case.get("maximum_absolute_error")
        relative_error = case.get("relative_l2_error")
        if (
            not isinstance(absolute_error, (int, float))
            or absolute_error > _MAXIMUM_ABSOLUTE_ERROR
            or not isinstance(relative_error, (int, float))
            or relative_error > _MAXIMUM_RELATIVE_L2_ERROR
        ):
            raise ValueError(f"SV-009 dispatch case {index} exceeds error tolerance")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    required_hosts: Sequence[str],
) -> dict[str, object]:
    """Combine the canonical dual-host/compiler public-dispatch matrix."""

    if tuple(required_hosts) != HOSTS:
        raise ValueError("SV-009 dispatch requires the canonical two hosts")
    for payload in payloads:
        validate_run(payload)
    expected = {(host, lane) for host in required_hosts for lane in COMPILER_LANES}
    observed = {
        (payload["host_label"], payload["compiler_lane"]) for payload in payloads
    }
    if len(payloads) != len(expected) or observed != expected:
        raise ValueError("SV-009 dispatch host/compiler matrix is incomplete")
    revisions = {payload["source_revision"] for payload in payloads}
    measurements = {
        json.dumps(payload["measurement"], sort_keys=True) for payload in payloads
    }
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("SV-009 dispatch runs must share revision and measurement")
    ordered = sorted(
        payloads,
        key=lambda item: (item["host_label"], item["compiler_lane"]),
    )
    cases = [case for run in ordered for case in run["cases"]]
    speedups = [float(case["public_speedup_over_pytorch"]) for case in cases]
    return {
        "benchmark": "statevector_local_2q_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "distribution_semantics": "single_device_public_runtime_dispatch",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device dispatch benchmark is not scalability evidence"
        ],
        "required_hosts": list(HOSTS),
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "public_speedup_range": [min(speedups), max(speedups)],
        "public_dispatch_win_on_all_runs": all(
            speedup >= PERFORMANCE_FLOOR for speedup in speedups
        ),
        "dispatch_selection_decision": "eligible_for_default",
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing a GPU provider."""

    expected = {
        "benchmark": "statevector_local_2q_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "distribution_semantics": "single_device_public_runtime_dispatch",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device dispatch benchmark is not scalability evidence"
        ],
        "required_hosts": list(HOSTS),
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "public_dispatch_win_on_all_runs": True,
        "dispatch_selection_decision": "eligible_for_default",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"SV-009 evidence field {field!r} is invalid")
    rebuilt = merge_runs(
        _sequence(payload.get("runs"), "runs"),
        required_hosts=HOSTS,
    )
    if dict(payload) != rebuilt:
        raise ValueError("SV-009 dispatch aggregate is not canonical")


def _read_payload(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="collect one host/compiler run")
    run.add_argument("--host-label", choices=HOSTS, required=True)
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261_009)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge", help="merge four raw runs")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--required-host", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an aggregate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        if args.warmup < 0 or args.repeats <= 0 or args.group_size <= 0:
            raise ValueError(
                "warmup must be non-negative; repeats and group_size must be positive"
            )
        payload = collect_run(args)
        validate_run(payload)
        _write_payload(args.output, payload)
    elif args.action == "merge":
        payload = merge_runs(
            [_read_payload(path) for path in args.input],
            required_hosts=args.required_host,
        )
        validate_evidence(payload)
        _write_payload(args.output, payload)
    else:
        validate_evidence(_read_payload(args.input))
        payload = {"validated": str(args.input)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
