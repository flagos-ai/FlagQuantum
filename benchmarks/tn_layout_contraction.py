"""Collect and validate NUM-002 layout-aware complex BMM evidence."""

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
from typing import Any, TypeAlias, TypedDict

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance
from flagquantum.simulation.real_imag_kernels import (
    _FUSED_LAYOUT_BMM_SHAPES,
    complex_einsum_pair,
)

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.tn_layout_contraction.v3"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.tn_layout_contraction.v3"
SEMANTIC_ID = "numerics.matmul.complex_batched_layout"
IMPLEMENTATION_ID = "FQKI-TRITON-NUM-002-A"
RUNNER = "benchmarks/tn_layout_contraction.py"
EQUATION = "azcb,czdb->zad"
SHAPE_MATRIX = (
    (16, 64, 16, 32, 64),
    (16, 128, 8, 16, 128),
    (16, 256, 8, 16, 256),
    (16, 64, 16, 64, 64),
)
COMPILER_LANES = ("stock_triton", "flagtree")
SELECTED_LOGICAL_BMM_SHAPES = _FUSED_LAYOUT_BMM_SHAPES
FALLBACK_MINIMUM_SPEEDUP = 0.95
FALLBACK_MAXIMUM_OVERHEAD_SECONDS = 5e-6
RESULT_NAMES = (
    "direct_layout_forward",
    "materialized_torch_bmm_forward",
    "native_einsum_forward",
    "public_catalog_dispatch",
    "direct_layout_forward_backward",
    "native_einsum_forward_backward",
)
MEASUREMENT_ORDERING = "balanced Latin square across all measured operations"

LayoutShapes: TypeAlias = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


class Measurement(TypedDict):
    """One synchronized timing and memory measurement."""

    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float
    peak_memory_bytes: int
    peak_memory_delta_bytes: int


def _inputs(
    batch: int,
    rows: int,
    reduction_left: int,
    reduction_right: int,
    columns: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, LayoutShapes]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    left = torch.randn(
        rows,
        batch,
        reduction_right,
        reduction_left,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    right = torch.randn(
        reduction_right,
        batch,
        columns,
        reduction_left,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    shapes = (
        (batch,),
        (rows,),
        (reduction_left, reduction_right),
        (columns,),
    )
    return left, right, shapes


def _direct(
    left: torch.Tensor,
    right: torch.Tensor,
    shapes: LayoutShapes,
) -> torch.Tensor:
    from flagquantum.kernels.triton.complex_bmm import fused_complex_layout_bmm

    return fused_complex_layout_bmm(left, right, (1, 0, 3, 2), (1, 3, 0, 2), shapes)


def _materialized(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    batch = left.shape[1]
    rows = left.shape[0]
    columns = right.shape[2]
    reduction = left.shape[2] * left.shape[3]
    return torch.bmm(
        left.permute(1, 0, 3, 2).reshape(batch, rows, reduction),
        right.permute(1, 3, 0, 2).reshape(batch, reduction, columns),
    )


def _public(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return complex_einsum_pair(EQUATION, left, right)


def _public_route(left: torch.Tensor, right: torch.Tensor) -> str:
    from flagquantum.simulation.real_imag_kernels import (
        _canonical_bmm_layout,
        _layout_bmm_dispatch_supported,
    )

    layout = _canonical_bmm_layout(EQUATION, left, right)
    if layout is not None and _layout_bmm_dispatch_supported(
        EQUATION, left, right, layout
    ):
        return "catalog_kernel"
    return "native_einsum"


def _counterbalanced_orders(
    names: Sequence[str], repeats: int
) -> tuple[tuple[str, ...], ...]:
    """Return a balanced Latin-square order for every measurement repeat."""

    if not names:
        raise ValueError("at least one measured operation is required")
    if len(set(names)) != len(names):
        raise ValueError("measured operation names must be unique")
    if repeats <= 0 or repeats % len(names) != 0:
        raise ValueError("repeats must be a positive multiple of operation count")
    count = len(names)
    if count % 2:
        raise ValueError(
            "balanced measurement ordering requires an even operation count"
        )
    first_row = tuple(
        0 if index == 0 else (index + 1) // 2 if index % 2 else count - index // 2
        for index in range(count)
    )
    return tuple(
        tuple(names[(item + repeat) % count] for item in first_row)
        for repeat in range(repeats)
    )


def _measure_group(function: Callable[[], object], group_size: int) -> float:
    started = time.perf_counter()
    for _ in range(group_size):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / group_size


def _measure_operations(
    operations: Mapping[str, Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, Measurement]:
    if tuple(operations) != RESULT_NAMES:
        raise ValueError("measurement operations must follow RESULT_NAMES")
    for _ in range(warmup):
        for function in operations.values():
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in operations}
    peak_memory: dict[str, list[int]] = {name: [] for name in operations}
    peak_deltas: dict[str, list[int]] = {name: [] for name in operations}
    for order in _counterbalanced_orders(tuple(operations), repeats):
        for name in order:
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            samples[name].append(_measure_group(operations[name], group_size))
            peak = torch.cuda.max_memory_allocated()
            peak_memory[name].append(peak)
            peak_deltas[name].append(max(0, peak - allocated_before))
    return {
        name: {
            "samples_seconds_per_invocation": samples[name],
            "median_seconds_per_invocation": statistics.median(samples[name]),
            "peak_memory_bytes": max(peak_memory[name]),
            "peak_memory_delta_bytes": max(peak_deltas[name]),
        }
        for name in operations
    }


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual - reference)
        / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).eps)
    )


def _case(
    batch: int,
    rows: int,
    reduction_left: int,
    reduction_right: int,
    columns: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    left, right, shapes = _inputs(
        batch,
        rows,
        reduction_left,
        reduction_right,
        columns,
        seed=seed,
    )
    direct = _direct(left, right, shapes)
    materialized = _materialized(left, right)
    native = torch.einsum(EQUATION, left, right)
    public = _public(left, right)
    for actual in (direct, materialized, public):
        torch.testing.assert_close(actual, native, rtol=5e-4, atol=2e-3)

    train_left = left.detach().requires_grad_(True)
    train_right = right.detach().requires_grad_(True)
    reference_left = left.detach().requires_grad_(True)
    reference_right = right.detach().requires_grad_(True)
    gradient_generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    output_gradient = torch.randn(
        batch,
        rows,
        columns,
        generator=gradient_generator,
        device="cuda",
        dtype=torch.complex64,
    )
    direct_gradients = torch.autograd.grad(
        _direct(train_left, train_right, shapes),
        (train_left, train_right),
        output_gradient,
    )
    native_gradients = torch.autograd.grad(
        torch.einsum(EQUATION, reference_left, reference_right),
        (reference_left, reference_right),
        output_gradient,
    )
    for actual, reference in zip(direct_gradients, native_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=5e-4, atol=2e-3)

    def direct_forward_backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            _direct(train_left, train_right, shapes),
            (train_left, train_right),
            output_gradient,
        )

    def native_forward_backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            torch.einsum(EQUATION, reference_left, reference_right),
            (reference_left, reference_right),
            output_gradient,
        )

    operations = {
        "direct_layout_forward": lambda: _direct(left, right, shapes),
        "materialized_torch_bmm_forward": lambda: _materialized(left, right),
        "native_einsum_forward": lambda: torch.einsum(EQUATION, left, right),
        "public_catalog_dispatch": lambda: complex_einsum_pair(EQUATION, left, right),
        "direct_layout_forward_backward": direct_forward_backward,
        "native_einsum_forward_backward": native_forward_backward,
    }
    results = _measure_operations(
        operations,
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    output_errors = [direct - native, materialized - native, public - native]
    gradient_errors = [
        actual - reference
        for actual, reference in zip(direct_gradients, native_gradients, strict=True)
    ]
    native_forward_seconds = float(
        results["native_einsum_forward"]["median_seconds_per_invocation"]
    )
    native_training_seconds = float(
        results["native_einsum_forward_backward"]["median_seconds_per_invocation"]
    )
    return {
        "shape": {
            "batch": batch,
            "rows": rows,
            "reduction_left": reduction_left,
            "reduction_right": reduction_right,
            "columns": columns,
        },
        "logical_bmm_shape": [
            batch,
            rows,
            reduction_left * reduction_right,
            columns,
        ],
        "dtype": "complex64",
        "layout": "explicit_strided_batch",
        "equation": EQUATION,
        "public_dispatch_route": _public_route(left, right),
        "maximum_absolute_error": max(
            float(torch.max(torch.abs(error))) for error in output_errors
        ),
        "relative_l2_error": max(
            _relative_l2(actual, native) for actual in (direct, materialized, public)
        ),
        "maximum_gradient_absolute_error": max(
            float(torch.max(torch.abs(error))) for error in gradient_errors
        ),
        "gradient_relative_l2_error": max(
            _relative_l2(actual, reference)
            for actual, reference in zip(
                direct_gradients, native_gradients, strict=True
            )
        ),
        **results,
        "direct_forward_speedup_over_native": (
            native_forward_seconds
            / float(results["direct_layout_forward"]["median_seconds_per_invocation"])
        ),
        "direct_forward_speedup_over_materialized": (
            float(
                results["materialized_torch_bmm_forward"][
                    "median_seconds_per_invocation"
                ]
            )
            / float(results["direct_layout_forward"]["median_seconds_per_invocation"])
        ),
        "public_dispatch_speedup_over_native": (
            native_forward_seconds
            / float(results["public_catalog_dispatch"]["median_seconds_per_invocation"])
        ),
        "direct_training_speedup_over_native": (
            native_training_seconds
            / float(
                results["direct_layout_forward_backward"][
                    "median_seconds_per_invocation"
                ]
            )
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed NUM-002 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NUM-002 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected_distribution = (
        "triton" if args.compiler_lane == "stock_triton" else "flagtree"
    )
    if identity_status != "resolved" or distribution != expected_distribution:
        raise RuntimeError(
            f"compiler lane {args.compiler_lane!r} requires distribution "
            f"{expected_distribution!r}, found {distribution!r} with status "
            f"{identity_status!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            *shape,
            seed=args.seed + index * 2,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, shape in enumerate(SHAPE_MATRIX)
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
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": args.compiler_lane,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "integration_path": integration_path,
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
            "ordering": MEASUREMENT_ORDERING,
            "order_cycle_length": len(RESULT_NAMES),
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


def _positive_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_run(payload: Mapping[str, Any]) -> None:
    """Fail closed when one raw run does not meet the evidence contract."""

    expected = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
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
    lane = payload.get("compiler_lane")
    if lane not in COMPILER_LANES:
        raise ValueError(f"unsupported compiler lane {lane!r}")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected_distribution = "triton" if lane == "stock_triton" else "flagtree"
    if (
        compiler.get("distribution") != expected_distribution
        or compiler.get("identity_status") != "resolved"
        or not compiler.get("version")
    ):
        raise ValueError(f"unresolved compiler identity for lane {lane!r}")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0 or repeats % len(RESULT_NAMES) != 0:
        raise ValueError(
            "measurement repeats must be a positive multiple of operation count"
        )
    if measurement.get("ordering") != MEASUREMENT_ORDERING:
        raise ValueError("measurement ordering must be the balanced Latin square")
    if measurement.get("order_cycle_length") != len(RESULT_NAMES):
        raise ValueError("measurement order_cycle_length is invalid")
    if not isinstance(measurement.get("warmup"), int) or measurement["warmup"] < 0:
        raise ValueError("measurement warmup must be a non-negative integer")
    if (
        not isinstance(measurement.get("group_size"), int)
        or measurement["group_size"] <= 0
    ):
        raise ValueError("measurement group_size must be a positive integer")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[Any, ...]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        shape_fields = (
            "batch",
            "rows",
            "reduction_left",
            "reduction_right",
            "columns",
        )
        shape_values = tuple(
            _positive_integer(shape.get(field), f"cases[{index}].shape.{field}")
            for field in shape_fields
        )
        observed_shapes.append(shape_values)
        batch, rows, reduction_left, reduction_right, columns = shape_values
        expected_logical_shape = (
            batch,
            rows,
            reduction_left * reduction_right,
            columns,
        )
        if tuple(case.get("logical_bmm_shape", ())) != expected_logical_shape:
            raise ValueError(f"cases[{index}] records the wrong logical BMM shape")
        if case.get("layout") != "explicit_strided_batch":
            raise ValueError(f"cases[{index}] records the wrong layout")
        if case.get("equation") != EQUATION:
            raise ValueError(f"cases[{index}] records the wrong equation")
        logical_shape = tuple(case["logical_bmm_shape"])
        expected_route = (
            "catalog_kernel"
            if logical_shape in SELECTED_LOGICAL_BMM_SHAPES
            else "native_einsum"
        )
        if case.get("public_dispatch_route") != expected_route:
            raise ValueError(
                f"cases[{index}] public dispatch route must equal {expected_route!r}"
            )
        for error_field, tolerance in (
            ("maximum_absolute_error", 2e-3),
            ("relative_l2_error", 5e-4),
            ("maximum_gradient_absolute_error", 2e-3),
            ("gradient_relative_l2_error", 5e-4),
        ):
            error = case.get(error_field)
            if not isinstance(error, (int, float)) or error < 0 or error > tolerance:
                raise ValueError(f"cases[{index}] exceeds {error_field} tolerance")
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds_per_invocation"),
                f"cases[{index}].{result_name}.samples_seconds_per_invocation",
            )
            if len(samples) != repeats:
                raise ValueError(
                    f"cases[{index}].{result_name} must contain {repeats} samples"
                )
            for sample_index, sample in enumerate(samples):
                _positive_number(
                    sample,
                    f"cases[{index}].{result_name}.samples[{sample_index}]",
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
        speedups = {
            "direct_forward_speedup_over_native": (
                case["native_einsum_forward"]["median_seconds_per_invocation"]
                / case["direct_layout_forward"]["median_seconds_per_invocation"]
            ),
            "direct_forward_speedup_over_materialized": (
                case["materialized_torch_bmm_forward"]["median_seconds_per_invocation"]
                / case["direct_layout_forward"]["median_seconds_per_invocation"]
            ),
            "public_dispatch_speedup_over_native": (
                case["native_einsum_forward"]["median_seconds_per_invocation"]
                / case["public_catalog_dispatch"]["median_seconds_per_invocation"]
            ),
            "direct_training_speedup_over_native": (
                case["native_einsum_forward_backward"]["median_seconds_per_invocation"]
                / case["direct_layout_forward_backward"][
                    "median_seconds_per_invocation"
                ]
            ),
        }
        for field, expected_speedup in speedups.items():
            if case.get(field) != expected_speedup:
                raise ValueError(f"cases[{index}] {field} is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed NUM-002 shape matrix")


def _fallback_within_overhead_budget(case: Mapping[str, Any]) -> bool:
    public_seconds = _positive_number(
        case["public_catalog_dispatch"]["median_seconds_per_invocation"],
        "public dispatch median",
    )
    native_seconds = _positive_number(
        case["native_einsum_forward"]["median_seconds_per_invocation"],
        "native einsum median",
    )
    public_speedup = _positive_number(
        case["public_dispatch_speedup_over_native"], "public dispatch speedup"
    )
    return (
        public_speedup >= FALLBACK_MINIMUM_SPEEDUP
        or public_seconds - native_seconds <= FALLBACK_MAXIMUM_OVERHEAD_SECONDS
    )


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the two-host, two-compiler matrix into one catalog artifact."""

    if len(set(required_hosts)) != len(required_hosts) or not required_hosts:
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    observed = {
        (payload["host_label"], payload["compiler_lane"]) for payload in payloads
    }
    required = {
        (host, compiler_lane)
        for host in required_hosts
        for compiler_lane in COMPILER_LANES
    }
    if observed != required or len(payloads) != len(required):
        raise ValueError(
            "runs must contain each required host and compiler lane exactly once"
        )
    ordered = sorted(
        payloads, key=lambda item: (item["host_label"], item["compiler_lane"])
    )
    direct_forward_wins = all(
        case["direct_forward_speedup_over_native"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    selected_kernel_wins = all(
        case["public_dispatch_speedup_over_native"] > 1.0
        for run in ordered
        for case in run["cases"]
        if case["public_dispatch_route"] == "catalog_kernel"
    )
    fallback_within_budget = all(
        _fallback_within_overhead_budget(case)
        for run in ordered
        for case in run["cases"]
        if case["public_dispatch_route"] == "native_einsum"
    )
    direct_training_wins = all(
        case["direct_training_speedup_over_native"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    return {
        "benchmark": "tn_layout_contraction",
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
            "single-device kernel benchmark is not distributed scalability evidence"
        ],
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "direct_forward_win_on_all_cases": direct_forward_wins,
        "selected_kernel_win_on_all_cases": selected_kernel_wins,
        "fallback_within_overhead_budget_on_all_cases": fallback_within_budget,
        "fallback_minimum_speedup": FALLBACK_MINIMUM_SPEEDUP,
        "fallback_maximum_overhead_seconds": FALLBACK_MAXIMUM_OVERHEAD_SECONDS,
        "direct_training_win_on_all_cases": direct_training_wins,
        "dispatch_evidence_decision": (
            "retain_current_policy"
            if selected_kernel_wins and fallback_within_budget
            else "revisit_current_policy"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "tn_layout_contraction",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
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
        "required_compiler_lanes": list(COMPILER_LANES),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence field {field!r} must equal {value!r}")
    hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=hosts)
    if dict(payload) != rebuilt:
        raise ValueError("aggregate is not the canonical merge of its raw runs")


def _read_payload(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="collect one host/compiler run")
    run.add_argument("--host-label", default=socket.gethostname())
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261003)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge", help="merge raw runs")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--required-host", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an aggregate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        if (
            args.warmup < 0
            or args.repeats <= 0
            or args.repeats % len(RESULT_NAMES) != 0
            or args.group_size <= 0
        ):
            raise ValueError(
                "warmup must be non-negative; repeats must be a positive multiple "
                "of operation count; group_size must be positive"
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
