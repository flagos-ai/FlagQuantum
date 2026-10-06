"""Benchmark the direct SV-011 wrapper against current product semantics."""

from __future__ import annotations

import argparse
import json
import platform
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

BENCHMARK = "statevector_pauli_rotation_2q"
RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_pauli_rotation_2q.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_pauli_rotation_2q.v1"
SEMANTIC_ID = "statevector.apply.pauli_rotation_2q.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-011-A"
RUNNER = "benchmarks/internal/evidence/statevector_pauli_rotation_2q_probe.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 16, (0, 1), "XX", False),
    (1, 20, (0, 19), "YY", False),
    (1, 20, (19, 0), "ZZ", False),
    (1, 24, (3, 19), "XX", False),
    (4, 20, (1, 18), "YY", True),
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
    qubits: tuple[int, int],
    pauli: str,
    batched_parameter: bool,
) -> dict[str, object]:
    return {
        "batch": batch,
        "n_qubits": n_qubits,
        "qubits": list(qubits),
        "pauli": pauli,
        "batched_parameter": batched_parameter,
    }


def _pauli_matrix(pauli: str, *, device: torch.device) -> torch.Tensor:
    one = torch.tensor(1.0, device=device, dtype=torch.complex64)
    zero = torch.tensor(0.0, device=device, dtype=torch.complex64)
    imaginary = torch.tensor(1.0j, device=device, dtype=torch.complex64)
    matrices = {
        "X": torch.stack((torch.stack((zero, one)), torch.stack((one, zero)))),
        "Y": torch.stack(
            (torch.stack((zero, -imaginary)), torch.stack((imaginary, zero)))
        ),
        "Z": torch.stack((torch.stack((one, zero)), torch.stack((zero, -one)))),
    }
    return torch.kron(matrices[pauli[0]], matrices[pauli[1]])


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed SV-011 matrix."""

    from flagquantum.kernels.provenance import triton_compiler_provenance
    from flagquantum.kernels.triton.statevector_pauli_rotation import (
        apply_complex64_local_pauli_rotation_2q,
    )
    from flagquantum.simulation.statevector.operations import (
        _apply_diagonal_matrix,
        _apply_matrix_layout,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-011 benchmark evidence")
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
    for index, shape in enumerate(SHAPE_MATRIX):
        batch, n_qubits, qubits, pauli, batched_parameter = shape
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        state = torch.randn(
            batch,
            1 << n_qubits,
            generator=generator,
            dtype=torch.complex64,
            device="cuda",
        )
        parameter_shape = (batch,) if batched_parameter else (1,)
        angles = torch.randn(
            parameter_shape,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        cosine = torch.cos(angles / 2)
        sine = torch.sin(angles / 2)
        rotation = torch.complex(cosine, sine)
        identity = torch.eye(4, device="cuda", dtype=torch.complex64)
        operator = _pauli_matrix(pauli, device=state.device)
        matrices = (
            cosine[:, None, None] * identity - 1.0j * sine[:, None, None] * operator
        )
        matrix = matrices[0] if not batched_parameter else matrices
        direct = partial(
            apply_complex64_local_pauli_rotation_2q,
            state,
            rotation,
            qubits=qubits,
            pauli=pauli,
        )
        reference = (
            partial(_apply_diagonal_matrix, state, matrix, qubits, n_qubits)
            if pauli == "ZZ"
            else partial(_apply_matrix_layout, state, matrix, qubits, n_qubits)
        )
        actual = direct()
        expected = reference()
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
                "shape": _shape_record(*shape),
                "direct_kernel_wrapper": direct_timing,
                "product_reference": reference_timing,
                "speedup_over_product": reference_timing[
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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("SV-011 evidence requires exactly four raw runs")
    identities = {(run.get("host_label"), run.get("compiler_lane")) for run in runs}
    if identities != {(host, lane) for host in HOSTS for lane in COMPILER_LANES}:
        raise ValueError("SV-011 host/compiler matrix is incomplete")
    revisions = {run.get("source_revision") for run in runs}
    measurements = {json.dumps(run.get("measurement"), sort_keys=True) for run in runs}
    expected_shapes = [_shape_record(*shape) for shape in SHAPE_MATRIX]
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("SV-011 evidence must share source and measurement policy")
    for run in runs:
        if run.get("schema") != RUN_SCHEMA or run.get("runner") != RUNNER:
            raise ValueError("SV-011 raw-run identity mismatch")
        cases = run.get("cases")
        if (
            not isinstance(cases, list)
            or [case.get("shape") for case in cases] != expected_shapes
        ):
            raise ValueError("SV-011 evidence shape matrix mismatch")
        for case_value in cases:
            case = _mapping(case_value, "case")
            if float(case.get("speedup_over_product", 0.0)) <= 0.0:
                raise ValueError("SV-011 evidence speedup must be positive")

    cases = [case for run in runs for case in run["cases"]]
    speedups = [float(case["speedup_over_product"]) for case in cases]
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
            "minimum_speedup_over_product": min(speedups),
            "maximum_speedup_over_product": max(speedups),
            "maximum_absolute_error": max(
                float(case["maximum_absolute_error"]) for case in cases
            ),
            "maximum_relative_l2_error": max(
                float(case["relative_l2_error"]) for case in cases
            ),
            "all_cases_win": all_cases_win,
            "decision": (
                "eligible_for_dispatch_evaluation"
                if all_cases_win
                else "optimize_before_promotion"
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
    run.add_argument("--seed", type=int, default=261_011)
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
