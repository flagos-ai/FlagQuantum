"""Collect and validate FlagTree TLE local one-qubit kernel evidence."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.flagtree_tle_local_1q.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.flagtree_tle_local_1q.v1"
SEMANTIC_ID = "statevector.apply.matrix_1q.local"
IMPLEMENTATION_ID = "FQKI-FLAGTREE-SV-001-A"
BASELINE_IMPLEMENTATION_ID = "FQKI-TRITON-SV-001-A"
RUNNER = "benchmarks/flagtree_tle_local_1q.py"
SHAPE_MATRIX = (
    (1, 1 << 10, 0),
    (1, 1 << 16, 8),
    (1, 1 << 20, 10),
    (1, 1 << 24, 12),
)
RESULT_NAMES = ("flagtree_tle", "shared_triton", "pytorch_reference")


def _reference_apply(
    state: torch.Tensor,
    matrix: torch.Tensor,
    bit_position: int,
) -> torch.Tensor:
    batch, size = state.shape
    high = size >> (bit_position + 1)
    paired = state.reshape(batch, high, 2, 1 << bit_position)
    return torch.einsum("ij,bhjw->bhiw", matrix, paired).reshape_as(state)


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(group_size):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / group_size)
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_delta_bytes": max(
            0,
            torch.cuda.max_memory_allocated() - allocated_before,
        ),
    }


def _case(
    batch: int,
    amplitudes: int,
    bit_position: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.flagtree import apply_complex64_local_1q_tle
    from flagquantum.kernels.triton.statevector_gates import apply_complex64_local_1q

    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        batch,
        amplitudes,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    matrix = torch.randn(
        2,
        2,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    tle_output = torch.empty_like(state)
    triton_output = torch.empty_like(state)
    reference = _reference_apply(state, matrix, bit_position)
    tle_actual = apply_complex64_local_1q_tle(
        state,
        matrix,
        bit_position=bit_position,
        output=tle_output,
    )
    triton_actual = apply_complex64_local_1q(
        state,
        matrix,
        bit_position=bit_position,
        output=triton_output,
    )
    torch.testing.assert_close(tle_actual, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(triton_actual, reference, atol=3e-5, rtol=3e-5)

    tle_result = _measure(
        lambda: apply_complex64_local_1q_tle(
            state,
            matrix,
            bit_position=bit_position,
            output=tle_output,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    triton_result = _measure(
        lambda: apply_complex64_local_1q(
            state,
            matrix,
            bit_position=bit_position,
            output=triton_output,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    pytorch_result = _measure(
        lambda: _reference_apply(state, matrix, bit_position),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    difference = tle_actual - reference
    tle_median = float(tle_result["median_seconds_per_invocation"])
    triton_median = float(triton_result["median_seconds_per_invocation"])
    pytorch_median = float(pytorch_result["median_seconds_per_invocation"])
    return {
        "shape": {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "bit_position": bit_position,
        },
        "dtype": "complex64",
        "layout": "contiguous_flat_statevector",
        "maximum_absolute_error": float(torch.max(torch.abs(difference))),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(
                torch.finfo(torch.float32).eps
            )
        ),
        "flagtree_tle": tle_result,
        "shared_triton": triton_result,
        "pytorch_reference": pytorch_result,
        "tle_speedup_over_shared_triton": triton_median / tle_median,
        "tle_speedup_over_pytorch": pytorch_median / tle_median,
        "shared_triton_speedup_over_pytorch": pytorch_median / triton_median,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed FlagTree-only matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlagTree TLE benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    if (
        identity_status != "resolved"
        or distribution != "flagtree"
        or integration_path != "flagtree"
    ):
        raise RuntimeError(
            "FlagTree TLE evidence requires the flagtree distribution to own "
            f"triton; found {distribution!r} with status {identity_status!r}"
        )

    import triton

    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            batch,
            amplitudes,
            bit_position,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (batch, amplitudes, bit_position) in enumerate(SHAPE_MATRIX)
    ]
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "baseline_implementation_id": BASELINE_IMPLEMENTATION_ID,
        "source_revision": args.source_revision,
        "runner": RUNNER,
        "command": shlex.join(sys.argv),
        "host_label": args.host_label,
        "reported_hostname": socket.gethostname(),
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "triton_api_version": triton.__version__,
            "integration_path": integration_path,
            "backend": triton.runtime.driver.active.get_current_target().backend,
            "identity_source": "python_package_metadata",
            "identity_status": identity_status,
        },
        "environment": {
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": args.group_size,
            "seed": args.seed,
        },
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
    """Fail closed when one raw run does not meet the evidence contract."""

    expected = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "baseline_implementation_id": BASELINE_IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    for field in ("source_revision", "command", "host_label"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"run field {field!r} must be a non-empty string")
    compiler = _mapping(payload.get("compiler"), "compiler")
    if (
        compiler.get("distribution") != "flagtree"
        or compiler.get("integration_path") != "flagtree"
        or compiler.get("backend") != "cuda"
        or compiler.get("identity_status") != "resolved"
        or not compiler.get("version")
    ):
        raise ValueError("run must record a resolved FlagTree CUDA compiler")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("measurement repeats must be a positive integer")
    if not isinstance(measurement.get("warmup"), int) or measurement["warmup"] < 0:
        raise ValueError("measurement warmup must be a non-negative integer")
    if (
        not isinstance(measurement.get("group_size"), int)
        or measurement["group_size"] <= 0
    ):
        raise ValueError("measurement group_size must be a positive integer")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[int, int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append(
            (
                shape.get("batch"),
                shape.get("amplitudes_per_batch"),
                shape.get("bit_position"),
            )
        )
        if case.get("maximum_absolute_error", float("inf")) > 3e-5:
            raise ValueError(f"cases[{index}] exceeds the absolute error tolerance")
        if case.get("relative_l2_error", float("inf")) > 3e-5:
            raise ValueError(f"cases[{index}] exceeds the relative error tolerance")
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds_per_invocation"),
                f"cases[{index}].{result_name}.samples_seconds_per_invocation",
            )
            if len(samples) != repeats or any(
                not isinstance(sample, (int, float)) or sample <= 0
                for sample in samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} must contain "
                    f"{repeats} positive samples"
                )
            if result.get("median_seconds_per_invocation") != statistics.median(
                samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} median is not reproducible"
                )
            for memory_field in ("peak_memory_bytes", "peak_memory_delta_bytes"):
                if (
                    not isinstance(result.get(memory_field), int)
                    or result[memory_field] < 0
                ):
                    raise ValueError(
                        f"cases[{index}].{result_name}.{memory_field} is invalid"
                    )
        tle_median = case["flagtree_tle"]["median_seconds_per_invocation"]
        triton_median = case["shared_triton"]["median_seconds_per_invocation"]
        pytorch_median = case["pytorch_reference"]["median_seconds_per_invocation"]
        expected_speedups = {
            "tle_speedup_over_shared_triton": triton_median / tle_median,
            "tle_speedup_over_pytorch": pytorch_median / tle_median,
            "shared_triton_speedup_over_pytorch": pytorch_median / triton_median,
        }
        for name, value in expected_speedups.items():
            if case.get(name) != value:
                raise ValueError(f"cases[{index}] {name} is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed FlagTree TLE shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    required_hosts: Sequence[str],
) -> dict[str, object]:
    """Combine the two-host FlagTree matrix into one catalog artifact."""

    if len(set(required_hosts)) != len(required_hosts) or not required_hosts:
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    observed_hosts = [str(payload["host_label"]) for payload in payloads]
    if sorted(observed_hosts) != sorted(required_hosts):
        raise ValueError("runs must contain each required host exactly once")
    ordered = sorted(payloads, key=lambda item: item["host_label"])
    tle_wins = all(
        case["tle_speedup_over_shared_triton"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    shared_triton_wins = all(
        case["shared_triton_speedup_over_pytorch"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    return {
        "benchmark": "flagtree_tle_local_1q",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "baseline_implementation_id": BASELINE_IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device kernel benchmark is not distributed scalability evidence"
        ],
        "required_hosts": sorted(required_hosts),
        "required_compiler_distribution": "flagtree",
        "tle_win_over_shared_triton_on_all_cases": tle_wins,
        "shared_triton_win_over_pytorch_on_all_cases": shared_triton_wins,
        "provider_selection_decision": (
            "eligible_for_dispatch_study" if tle_wins else "retain_explicit"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing FlagTree or Triton."""

    expected = {
        "benchmark": "flagtree_tle_local_1q",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "baseline_implementation_id": BASELINE_IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "required_compiler_distribution": "flagtree",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence field {field!r} must equal {value!r}")
    required_hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=required_hosts)
    for field in (
        "source_revision",
        "tle_win_over_shared_triton_on_all_cases",
        "shared_triton_win_over_pytorch_on_all_cases",
        "provider_selection_decision",
    ):
        if payload.get(field) != rebuilt[field]:
            raise ValueError(f"evidence field {field!r} is not reproducible")
    if not payload.get("scalability_blockers"):
        raise ValueError("local evidence must name its scalability blocker")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--host-label", required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261006)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--required-host", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        payload = collect_run(args)
        validate_run(payload)
        _write_json(args.output, payload)
    elif args.command == "merge":
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
        payload = merge_runs(payloads, required_hosts=args.required_host)
        validate_evidence(payload)
        _write_json(args.output, payload)
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        validate_evidence(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
