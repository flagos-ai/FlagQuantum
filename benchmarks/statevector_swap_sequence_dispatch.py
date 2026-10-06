"""Collect two-host public-dispatch evidence for SV-014."""

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

BENCHMARK = "statevector_swap_sequence_dispatch"
RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_swap_sequence_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_swap_sequence_dispatch.v1"
SEMANTIC_ID = "statevector.apply.swap_sequence.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-014-A"
RUNNER = "benchmarks/statevector_swap_sequence_dispatch.py"
DISPATCH_VARIABLE = "FQ_TRITON_SWAP_SEQUENCE"
PERFORMANCE_FLOOR = 1.0
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 1 << 16, ((0, 15), (1, 14), (2, 13), (3, 12))),
    (1, 1 << 20, ((0, 19), (1, 18), (2, 17), (3, 16))),
    (1, 1 << 20, tuple((index, 19 - index) for index in range(6))),
    (1, 1 << 24, tuple((index, 23 - index) for index in range(8))),
    (4, 1 << 20, tuple((index, 19 - index) for index in range(5))),
)
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
_RESULT_NAMES = ("public_catalog_dispatch", "public_pytorch_reference")


class TimingResult(TypedDict):
    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float


def _reference(
    state: torch.Tensor,
    swaps: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    from flagquantum.simulation.statevector.operations import _apply_fixed_permutation

    output = state
    n_qubits = state.shape[1].bit_length() - 1
    for qubits in swaps:
        output = _apply_fixed_permutation(output, "swap", qubits, n_qubits)
    return output


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
    amplitudes: int,
    swaps: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "batch": batch,
        "amplitudes_per_batch": amplitudes,
        "swaps": [list(pair) for pair in swaps],
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed SV-014 dispatch matrix."""

    from flagquantum.kernels.provenance import triton_compiler_provenance
    from flagquantum.simulation.statevector.swap_sequence_dispatch import (
        _swap_sequence_kernel_enabled,
        _try_apply_cataloged_swap_sequence,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-014 dispatch evidence")
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
    for index, shape in enumerate(SHAPE_MATRIX):
        batch, amplitudes, swaps = shape
        n_qubits = amplitudes.bit_length() - 1
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            amplitudes,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        if not _swap_sequence_kernel_enabled(
            state,
            swaps=swaps,
            n_qubits=n_qubits,
        ):
            raise RuntimeError("the fixed SV-014 case did not select the kernel")
        dispatch = partial(
            _try_apply_cataloged_swap_sequence,
            state,
            swaps=swaps,
            n_qubits=n_qubits,
        )
        reference = partial(_reference, state, swaps)
        actual = dispatch()
        if actual is None:
            raise RuntimeError("the fixed SV-014 dispatch unexpectedly fell back")
        expected = reference()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
                "shape": _shape_record(*shape),
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
        "seed": args.seed,
        "cases": cases,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def validate_run(run: Mapping[str, Any]) -> None:
    """Fail closed when one raw public-dispatch run misses its contract."""

    expected_fields = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }
    for field, expected in expected_fields.items():
        if run.get(field) != expected:
            raise ValueError(f"SV-014 dispatch field {field!r} is invalid")
    if _FULL_REVISION.fullmatch(str(run.get("source_revision"))) is None:
        raise ValueError("SV-014 dispatch source revision must be a full Git hash")
    host = run.get("host_label")
    lane = run.get("compiler_lane")
    if host not in HOSTS or lane not in COMPILER_LANES:
        raise ValueError("SV-014 dispatch host or compiler lane is invalid")
    compiler = _mapping(run.get("compiler"), "compiler")
    expected_compiler = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[str(lane)]
    if (
        compiler.get("distribution"),
        compiler.get("integration_path"),
    ) != expected_compiler or compiler.get("identity_status") != "resolved":
        raise ValueError("SV-014 dispatch compiler identity does not match its lane")
    measurement = _mapping(run.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SV-014 dispatch repeats must be positive")
    expected_shapes = [_shape_record(*shape) for shape in SHAPE_MATRIX]
    cases = run.get("cases")
    if (
        not isinstance(cases, list)
        or [case.get("shape") for case in cases] != expected_shapes
    ):
        raise ValueError("SV-014 dispatch evidence shape matrix mismatch")
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"case {index}")
        medians = {}
        for result_name in _RESULT_NAMES:
            result = _mapping(case.get(result_name), f"case {index} result")
            samples = result.get("samples_seconds_per_invocation")
            if (
                not isinstance(samples, list)
                or len(samples) != repeats
                or any(
                    not isinstance(value, (int, float)) or value <= 0
                    for value in samples
                )
            ):
                raise ValueError(f"SV-014 dispatch case {index} samples are invalid")
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"SV-014 dispatch case {index} median is invalid")
            medians[result_name] = median
        speedup = (
            medians["public_pytorch_reference"] / medians["public_catalog_dispatch"]
        )
        if case.get("public_speedup_over_pytorch") != speedup:
            raise ValueError(f"SV-014 dispatch case {index} speedup is invalid")
        if speedup < PERFORMANCE_FLOOR:
            raise ValueError("SV-014 dispatch case misses performance floor")
        if float(case.get("maximum_absolute_error", float("inf"))) != 0.0:
            raise ValueError("SV-014 dispatch must be bitwise exact")
        if float(case.get("relative_l2_error", float("inf"))) != 0.0:
            raise ValueError("SV-014 dispatch must have zero relative error")


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-014 dispatch evidence requires exactly four raw runs")
    for run in runs:
        validate_run(run)
    identities = {(run["host_label"], run["compiler_lane"]) for run in runs}
    if identities != {(host, lane) for host in HOSTS for lane in COMPILER_LANES}:
        raise ValueError("SV-014 dispatch host/compiler matrix is incomplete")
    revisions = {run["source_revision"] for run in runs}
    measurements = {json.dumps(run["measurement"], sort_keys=True) for run in runs}
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError(
            "SV-014 dispatch evidence must share source and measurement policy"
        )
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
        "performance_floor": PERFORMANCE_FLOOR,
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
            "decision": "default_dispatch_enabled",
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
    run.add_argument("--seed", type=int, default=261_014)
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
