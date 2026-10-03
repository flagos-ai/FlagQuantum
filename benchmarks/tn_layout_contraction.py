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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.tn_layout_contraction.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.tn_layout_contraction.v1"
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
RESULT_NAMES = (
    "direct_layout_forward",
    "materialized_torch_bmm_forward",
    "native_einsum_forward",
    "public_catalog_dispatch",
    "direct_layout_forward_backward",
    "native_einsum_forward_backward",
)

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
    from flagquantum.simulation.real_imag_kernels import complex_einsum_pair

    return complex_einsum_pair(EQUATION, left, right)


def _measure(
    function: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> Measurement:
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

    results = {
        "direct_layout_forward": _measure(
            lambda: _direct(left, right, shapes),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "materialized_torch_bmm_forward": _measure(
            lambda: _materialized(left, right),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "native_einsum_forward": _measure(
            lambda: torch.einsum(EQUATION, left, right),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "public_catalog_dispatch": _measure(
            lambda: _public(left, right),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "direct_layout_forward_backward": _measure(
            direct_forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "native_einsum_forward_backward": _measure(
            native_forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
    }
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
    observed_shapes: list[tuple[Any, ...]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append(
            tuple(
                shape.get(field)
                for field in (
                    "batch",
                    "rows",
                    "reduction_left",
                    "reduction_right",
                    "columns",
                )
            )
        )
        if case.get("layout") != "explicit_strided_batch":
            raise ValueError(f"cases[{index}] records the wrong layout")
        if case.get("equation") != EQUATION:
            raise ValueError(f"cases[{index}] records the wrong equation")
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
    public_wins = all(
        case["public_dispatch_speedup_over_native"] > 1.0
        for run in ordered
        for case in run["cases"]
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
        "public_dispatch_win_on_all_cases": public_wins,
        "direct_training_win_on_all_cases": direct_training_wins,
        "dispatch_evidence_decision": (
            "retain_current_policy" if public_wins else "revisit_current_policy"
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
