"""Collect and validate FlagTree TLE distributed transpose-gate evidence."""

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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.flagtree_tle_transpose_1q.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.flagtree_tle_transpose_1q.v1"
SEMANTIC_ID = "statevector.distributed.transpose_apply_1q"
IMPLEMENTATION_ID = "FQKI-FLAGTREE-SV-006-A"
BASELINE_IMPLEMENTATION_ID = "FQKI-TRITON-SV-006-A"
RUNNER = "benchmarks/flagtree_tle_transpose_1q.py"
SHAPE_MATRIX = (
    (1, 1 << 10, 0, 0),
    (1, 1 << 16, 8, 1),
    (1, 1 << 20, 10, 0),
    (1, 1 << 24, 12, 1),
)
RESULT_NAMES = ("flagtree_tle", "shared_triton", "pytorch_reference")


def _reference_apply(
    state: torch.Tensor,
    received: torch.Tensor,
    matrix: torch.Tensor,
    bit_position: int,
    exchanged_bit_value: int,
) -> torch.Tensor:
    batch = state.shape[0]
    retained = state.reshape(batch, -1, 2, 1 << bit_position)[
        :, :, 1 - exchanged_bit_value, :
    ]
    remote = received.reshape_as(retained)
    pair = (
        torch.stack((remote, retained), dim=2)
        if exchanged_bit_value == 0
        else torch.stack((retained, remote), dim=2)
    )
    return torch.einsum("ij,bpjw->bpiw", matrix, pair).reshape_as(state)


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
            0, torch.cuda.max_memory_allocated() - allocated_before
        ),
    }


def _case(
    batch: int,
    amplitudes: int,
    bit_position: int,
    exchanged_bit_value: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.flagtree import (
        apply_complex64_transpose_1q_tle_inplace,
    )
    from flagquantum.kernels.triton.statevector_gates import (
        apply_complex64_transpose_1q_inplace,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        batch,
        amplitudes,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    received = torch.randn(
        batch,
        amplitudes // 2,
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
    expected = _reference_apply(
        state, received, matrix, bit_position, exchanged_bit_value
    )
    tle_state = state.clone()
    shared_state = state.clone()
    tle_actual = apply_complex64_transpose_1q_tle_inplace(
        tle_state,
        received,
        matrix,
        bit_position=bit_position,
        exchanged_bit_value=exchanged_bit_value,
    )
    shared_actual = apply_complex64_transpose_1q_inplace(
        shared_state,
        received,
        matrix,
        bit_position=bit_position,
        exchanged_bit_value=exchanged_bit_value,
    )
    torch.testing.assert_close(tle_actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(shared_actual, expected, atol=3e-5, rtol=3e-5)
    difference = tle_actual - expected
    maximum_absolute_error = float(torch.max(torch.abs(difference)))
    relative_l2_error = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected).clamp_min(torch.finfo(torch.float32).eps)
    )

    tle_state.copy_(state)
    shared_state.copy_(state)
    tle_result = _measure(
        lambda: apply_complex64_transpose_1q_tle_inplace(
            tle_state,
            received,
            matrix,
            bit_position=bit_position,
            exchanged_bit_value=exchanged_bit_value,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    shared_result = _measure(
        lambda: apply_complex64_transpose_1q_inplace(
            shared_state,
            received,
            matrix,
            bit_position=bit_position,
            exchanged_bit_value=exchanged_bit_value,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    reference_result = _measure(
        lambda: _reference_apply(
            state, received, matrix, bit_position, exchanged_bit_value
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    tle_median = float(tle_result["median_seconds_per_invocation"])
    shared_median = float(shared_result["median_seconds_per_invocation"])
    reference_median = float(reference_result["median_seconds_per_invocation"])
    return {
        "shape": {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "bit_position": bit_position,
            "exchanged_bit_value": exchanged_bit_value,
        },
        "dtype": "complex64",
        "layout": "contiguous_sharded_statevector",
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative_l2_error,
        "flagtree_tle": tle_result,
        "shared_triton": shared_result,
        "pytorch_reference": reference_result,
        "tle_speedup_over_shared_triton": shared_median / tle_median,
        "tle_speedup_over_pytorch": reference_median / tle_median,
        "shared_triton_speedup_over_pytorch": reference_median / shared_median,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed FlagTree-only matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlagTree TLE benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    if (distribution, integration_path, identity_status) != (
        "flagtree",
        "flagtree",
        "resolved",
    ):
        raise RuntimeError("FlagTree must own the active Triton CUDA compiler")
    import triton

    properties = torch.cuda.get_device_properties(0)
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
        "cases": [
            _case(
                *shape,
                seed=args.seed + index,
                warmup=args.warmup,
                repeats=args.repeats,
                group_size=args.group_size,
            )
            for index, shape in enumerate(SHAPE_MATRIX)
        ],
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
    observed_shapes: list[tuple[int, int, int, int]] = []
    for case_index, case_value in enumerate(_sequence(payload.get("cases"), "cases")):
        case = _mapping(case_value, f"cases[{case_index}]")
        shape = _mapping(case.get("shape"), f"cases[{case_index}].shape")
        observed_shapes.append(
            tuple(
                shape[field]
                for field in (
                    "batch",
                    "amplitudes_per_batch",
                    "bit_position",
                    "exchanged_bit_value",
                )
            )
        )
        if case.get("maximum_absolute_error", float("inf")) > 3e-5:
            raise ValueError(f"cases[{case_index}] exceeds error tolerance")
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), result_name)
            samples = _sequence(result.get("samples_seconds_per_invocation"), "samples")
            if len(samples) != repeats or any(
                not isinstance(sample, (int, float)) or sample <= 0
                for sample in samples
            ):
                raise ValueError("each result must contain positive timed samples")
            if result.get("median_seconds_per_invocation") != statistics.median(
                samples
            ):
                raise ValueError("result median is not reproducible")
        tle = case["flagtree_tle"]["median_seconds_per_invocation"]
        shared = case["shared_triton"]["median_seconds_per_invocation"]
        reference = case["pytorch_reference"]["median_seconds_per_invocation"]
        ratios = {
            "tle_speedup_over_shared_triton": shared / tle,
            "tle_speedup_over_pytorch": reference / tle,
            "shared_triton_speedup_over_pytorch": reference / shared,
        }
        if any(case.get(name) != value for name, value in ratios.items()):
            raise ValueError("case speedup is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the two-host matrix into one catalog artifact."""

    if len(set(required_hosts)) != len(required_hosts) or not required_hosts:
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    if sorted(str(payload["host_label"]) for payload in payloads) != sorted(
        required_hosts
    ):
        raise ValueError("runs must cover each required host exactly once")
    ordered = sorted(payloads, key=lambda item: str(item["host_label"]))
    tle_wins = all(
        case["tle_speedup_over_shared_triton"] > 1.0
        for payload in ordered
        for case in payload["cases"]
    )
    shared_wins = all(
        case["shared_triton_speedup_over_pytorch"] > 1.0
        for payload in ordered
        for case in payload["cases"]
    )
    return {
        "benchmark": "flagtree_tle_transpose_1q",
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
        "shared_triton_win_over_pytorch_on_all_cases": shared_wins,
        "provider_selection_decision": (
            "eligible_for_dispatch_study" if tle_wins else "retain_explicit"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing FlagTree or Triton."""

    expected = {
        "benchmark": "flagtree_tle_transpose_1q",
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
