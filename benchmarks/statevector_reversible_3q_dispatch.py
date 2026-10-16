"""Collect two-host public-dispatch evidence for SV-013."""

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

BENCHMARK = "statevector_reversible_3q_dispatch"
RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_reversible_3q_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_reversible_3q_dispatch.v1"
SEMANTIC_ID = "statevector.apply.reversible_permutation_3q.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-013-A"
RUNNER = "benchmarks/statevector_reversible_3q_dispatch.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_REVERSIBLE_3Q"
PERFORMANCE_FLOOR = 1.0
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
SHAPE_MATRIX = (
    (1, 20, (0, 1, 19), "ccx"),
    (1, 20, (17, 3, 11), "cswap"),
    (1, 24, (3, 19, 7), "ccx"),
    (4, 20, (1, 18, 7), "cswap"),
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
    qubits: tuple[int, int, int],
    opcode: str,
) -> dict[str, object]:
    return {
        "batch": batch,
        "n_qubits": n_qubits,
        "qubits": list(qubits),
        "opcode": opcode,
    }


def _apply_product_reference(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int, int],
    n_qubits: int,
    opcode: str,
) -> torch.Tensor:
    """Apply the existing dense eight-by-eight route."""

    from flagquantum.simulation.matrices import FREDKIN_MATRIX, TOFFOLI_MATRIX
    from flagquantum.simulation.statevector.operations import _apply_matrix_layout

    matrix = (TOFFOLI_MATRIX if opcode == "ccx" else FREDKIN_MATRIX).to(
        device=state.device,
        dtype=state.dtype,
    )
    return _apply_matrix_layout(state, matrix, qubits, n_qubits)


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed SV-013 dispatch matrix."""

    from flagquantum.kernels.provenance import triton_compiler_provenance
    from flagquantum.simulation.statevector.reversible_3q_dispatch import (
        _reversible_3q_kernel_enabled,
        _try_apply_cataloged_reversible_3q,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-013 dispatch evidence")
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
        batch, n_qubits, qubits, opcode = shape
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            1 << n_qubits,
            generator=generator,
            dtype=torch.complex64,
            device="cuda",
        )
        if not _reversible_3q_kernel_enabled(
            state,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        ):
            raise RuntimeError("the fixed SV-013 case did not select the kernel")
        dispatch = partial(
            _try_apply_cataloged_reversible_3q,
            state,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        reference = partial(
            _apply_product_reference,
            state,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        actual = dispatch()
        if actual is None:
            raise RuntimeError("the fixed SV-013 dispatch unexpectedly fell back")
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


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-013 dispatch evidence requires exactly four raw runs")
    identities = {(run.get("host_label"), run.get("compiler_lane")) for run in runs}
    if identities != {(host, lane) for host in HOSTS for lane in COMPILER_LANES}:
        raise ValueError("SV-013 dispatch host/compiler matrix is incomplete")
    revisions = {run.get("source_revision") for run in runs}
    measurements = {json.dumps(run.get("measurement"), sort_keys=True) for run in runs}
    expected_shapes = [_shape_record(*shape) for shape in SHAPE_MATRIX]
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError(
            "SV-013 dispatch evidence must share source and measurement policy"
        )
    for run in runs:
        if run.get("schema") != RUN_SCHEMA or run.get("runner") != RUNNER:
            raise ValueError("SV-013 dispatch raw-run identity mismatch")
        if run.get("semantic_id") != SEMANTIC_ID:
            raise ValueError("SV-013 dispatch semantic mismatch")
        if run.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError("SV-013 dispatch implementation mismatch")
        if _FULL_REVISION.fullmatch(str(run.get("source_revision"))) is None:
            raise ValueError("SV-013 dispatch source revision must be a full Git hash")
        cases = run.get("cases")
        if (
            not isinstance(cases, list)
            or [case.get("shape") for case in cases] != expected_shapes
        ):
            raise ValueError("SV-013 dispatch evidence shape matrix mismatch")
        for case_value in cases:
            case = _mapping(case_value, "case")
            if float(case.get("public_speedup_over_pytorch", 0.0)) < PERFORMANCE_FLOOR:
                raise ValueError("SV-013 dispatch case misses performance floor")
            if float(case.get("maximum_absolute_error", float("inf"))) != 0.0:
                raise ValueError("SV-013 dispatch must be bitwise exact")
            if float(case.get("relative_l2_error", float("inf"))) != 0.0:
                raise ValueError("SV-013 dispatch must have zero relative error")

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
        "shape_matrix": expected_shapes,
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
    run.add_argument("--seed", type=int, default=261_013)
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
