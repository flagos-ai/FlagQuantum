"""Collect two-host, two-compiler dispatch evidence for SV-003."""

from __future__ import annotations

import argparse
import json
import re
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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_local_cx_segment_dispatch.v1"
EVIDENCE_SCHEMA = (
    "flagquantum.kernel_benchmark.statevector_local_cx_segment_dispatch.v1"
)
SEMANTIC_ID = "statevector.apply.cnot_sequence.local"
IMPLEMENTATION_ID = "FQKI-TRITON-SV-003-A"
RUNNER = "benchmarks/statevector_local_cx_segment_dispatch.py"
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 1 << 20, (19, 18, 17, 16), (10, 11, 12, 13), False),
    (1, 1 << 24, (23, 22), (12, 13), True),
    (1, 1 << 24, (23, 22, 21, 20), (12, 13, 14, 15), True),
    (
        1,
        1 << 24,
        (23, 22, 21, 20, 19, 18, 17, 16),
        (12, 13, 14, 15, 16, 17, 18, 19),
        True,
    ),
)
RESULT_NAMES = ("catalog_dispatch", "pytorch_reference")
PERFORMANCE_FLOOR = 1.0
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


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
    }


def _reference_indices(
    amplitudes: int,
    control_bit_positions: tuple[int, ...],
    target_bit_positions: tuple[int, ...],
) -> torch.Tensor:
    indices = torch.arange(amplitudes, device="cuda")
    for control, target in reversed(
        tuple(zip(control_bit_positions, target_bit_positions, strict=True))
    ):
        indices ^= ((indices >> control) & 1) << target
    return indices


def _median(result: Mapping[str, object]) -> float:
    value = result.get("median_seconds_per_invocation")
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark median must be numeric")
    return float(value)


def _case(
    batch: int,
    amplitudes: int,
    control_bit_positions: tuple[int, ...],
    target_bit_positions: tuple[int, ...],
    default_eligible: bool,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.statevector_gates import (
        apply_complex64_local_cx_segment,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(
        batch,
        amplitudes,
        device="cuda",
        dtype=torch.complex64,
        generator=generator,
    )
    source = _reference_indices(amplitudes, control_bit_positions, target_bit_positions)
    catalog_output = torch.empty_like(base)
    reference_output = torch.empty_like(base)

    def catalog_dispatch() -> torch.Tensor:
        return apply_complex64_local_cx_segment(
            base,
            control_bit_positions=control_bit_positions,
            target_bit_positions=target_bit_positions,
            output=catalog_output,
        )

    def pytorch_reference() -> torch.Tensor:
        torch.index_select(base, 1, source, out=reference_output)
        return reference_output

    actual = catalog_dispatch().clone()
    expected = pytorch_reference().clone()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    catalog_result = _measure(
        catalog_dispatch,
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    reference_result = _measure(
        pytorch_reference,
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    catalog_median = _median(catalog_result)
    reference_median = _median(reference_result)
    return {
        "shape": {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "control_bit_positions": list(control_bit_positions),
            "target_bit_positions": list(target_bit_positions),
            "sequence_length": len(control_bit_positions),
        },
        "dtype": "complex64",
        "layout": "contiguous_flat_statevector",
        "default_eligible": default_eligible,
        "maximum_absolute_error": float(torch.max(torch.abs(actual - expected))),
        "catalog_dispatch": catalog_result,
        "pytorch_reference": reference_result,
        "speedup_over_pytorch": reference_median / catalog_median,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one fixed compiler lane and return its raw measurements."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SV-003 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[args.compiler_lane]
    if (
        identity_status != "resolved"
        or distribution != expected[0]
        or integration_path != expected[1]
    ):
        raise RuntimeError(
            f"{args.compiler_lane} requires compiler identity {expected!r}; "
            f"found {(distribution, integration_path)!r}"
        )
    import triton

    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            batch,
            amplitudes,
            controls,
            targets,
            default_eligible,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (
            batch,
            amplitudes,
            controls,
            targets,
            default_eligible,
        ) in enumerate(SHAPE_MATRIX)
    ]
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": args.source_revision,
        "runner": RUNNER,
        "command": shlex.join(sys.argv),
        "host_label": args.host_label,
        "reported_hostname": socket.gethostname(),
        "compiler_lane": args.compiler_lane,
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
    """Fail closed when one raw run violates the fixed evidence contract."""

    for field, value in {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or _FULL_REVISION.fullmatch(revision) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    lane = payload.get("compiler_lane")
    if lane not in COMPILER_LANES:
        raise ValueError("compiler_lane is not recognized")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[lane]
    if (
        compiler.get("distribution") != expected[0]
        or compiler.get("integration_path") != expected[1]
        or compiler.get("backend") != "cuda"
        or compiler.get("identity_status") != "resolved"
    ):
        raise ValueError("run compiler identity does not match its lane")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("measurement repeats must be positive")
    observed = []
    for index, case_value in enumerate(_sequence(payload.get("cases"), "cases")):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed.append(
            (
                shape.get("batch"),
                shape.get("amplitudes_per_batch"),
                tuple(shape.get("control_bit_positions", ())),
                tuple(shape.get("target_bit_positions", ())),
                case.get("default_eligible"),
            )
        )
        if case.get("maximum_absolute_error") != 0.0:
            raise ValueError(f"cases[{index}] must be bitwise exact")
        medians = {}
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
                raise ValueError(f"cases[{index}].{result_name} samples are invalid")
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"cases[{index}].{result_name} median is invalid")
            medians[result_name] = median
        speedup = medians["pytorch_reference"] / medians["catalog_dispatch"]
        if case.get("speedup_over_pytorch") != speedup:
            raise ValueError(f"cases[{index}] speedup is not reproducible")
        if case.get("default_eligible") and speedup < PERFORMANCE_FLOOR:
            raise ValueError(f"cases[{index}] misses the default performance floor")
    if tuple(observed) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed SV-003 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the required host and compiler matrix into one artifact."""

    if not required_hosts or len(set(required_hosts)) != len(required_hosts):
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record one source revision")
    observed = {(run["host_label"], run["compiler_lane"]) for run in payloads}
    expected = {(host, lane) for host in required_hosts for lane in COMPILER_LANES}
    if observed != expected or len(payloads) != len(expected):
        raise ValueError("runs must contain every host/compiler lane exactly once")
    ordered = sorted(
        payloads, key=lambda item: (item["host_label"], item["compiler_lane"])
    )
    eligible_speedups = [
        float(case["speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
        if case["default_eligible"]
    ]
    return {
        "benchmark": "statevector_local_cx_segment_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
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
            "single-device kernel evidence is not distributed scalability evidence"
        ],
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "minimum_default_window_speedup": min(eligible_speedups),
        "default_window_passed": all(
            speedup >= PERFORMANCE_FLOOR for speedup in eligible_speedups
        ),
        "dispatch_decision": "eligible_for_default",
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing an accelerator provider."""

    for field, value in {
        "benchmark": "statevector_local_cx_segment_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "dispatch_decision": "eligible_for_default",
    }.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence field {field!r} must equal {value!r}")
    required_hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=required_hosts)
    for field in (
        "source_revision",
        "minimum_default_window_speedup",
        "default_window_passed",
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
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261004)
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
