"""Collect and validate NUM-001 complex BMM evidence on one CUDA device."""

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
from typing import Any, cast

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.complex_bmm_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.complex_bmm_dispatch.v1"
SEMANTIC_ID = "numerics.matmul.complex_batched"
IMPLEMENTATION_ID = "FQKI-TRITON-NUM-001-A"
RUNNER = "benchmarks/complex_bmm_dispatch.py"
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 32, 32, 64, "right_going_transfer"),
    (1, 64, 32, 32, "left_going_transfer"),
    (16, 32, 32, 64, "right_going_transfer"),
    (16, 64, 32, 32, "left_going_transfer"),
    (16, 64, 64, 128, "right_going_transfer"),
    (16, 128, 64, 64, "left_going_transfer"),
    (64, 64, 64, 128, "right_going_transfer"),
    (64, 128, 64, 64, "left_going_transfer"),
    (16, 128, 128, 256, "right_going_transfer"),
    (16, 256, 128, 128, "left_going_transfer"),
)
RESULT_NAMES = (
    "pytorch_bmm_forward",
    "triton_forward",
    "pytorch_bmm_forward_backward",
    "triton_forward_backward",
)


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def _measure_group(operation: Callable[[], object], group_size: int) -> float:
    started = time.perf_counter()
    for _ in range(group_size):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / group_size


def _measure_pair(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    peaks: dict[str, list[int]] = {"baseline": [], "candidate": []}
    operations = {"baseline": baseline, "candidate": candidate}
    for repeat in range(repeats):
        order = (
            ("baseline", "candidate")
            if repeat % 2 == 0
            else (
                "candidate",
                "baseline",
            )
        )
        for name in order:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            samples[name].append(_measure_group(operations[name], group_size))
            peaks[name].append(max(0, torch.cuda.max_memory_allocated() - before))

    def result(name: str) -> dict[str, object]:
        return {
            "median_seconds_per_invocation": statistics.median(samples[name]),
            "samples_seconds_per_invocation": samples[name],
            "peak_memory_delta_bytes": max(peaks[name]),
        }

    return result("baseline"), result("candidate")


def _case(
    batch: int,
    rows: int,
    reduction: int,
    columns: int,
    direction: str,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton import fused_complex_bmm

    generator = torch.Generator(device="cuda").manual_seed(seed)
    left = torch.randn(
        batch,
        rows,
        reduction,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    right = torch.randn(
        batch,
        reduction,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    output_gradient = torch.randn(
        batch,
        rows,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    inference_left, inference_right = left.detach(), right.detach()

    def pytorch_forward() -> torch.Tensor:
        return torch.bmm(inference_left, inference_right)

    def triton_forward() -> torch.Tensor:
        return fused_complex_bmm(inference_left, inference_right)

    def pytorch_forward_backward() -> tuple[torch.Tensor, ...]:
        output = torch.bmm(left, right)
        return torch.autograd.grad(output, (left, right), output_gradient)

    def triton_forward_backward() -> tuple[torch.Tensor, ...]:
        output = fused_complex_bmm(left, right)
        return torch.autograd.grad(output, (left, right), output_gradient)

    reference = pytorch_forward()
    actual = triton_forward()
    reference_gradients = pytorch_forward_backward()
    actual_gradients = triton_forward_backward()
    forward_error = float(torch.max(torch.abs(actual - reference)).detach().cpu())
    gradient_errors = [
        float(torch.max(torch.abs(value - expected)).detach().cpu())
        for value, expected in zip(actual_gradients, reference_gradients, strict=True)
    ]
    gradient_relative_errors = [
        _relative_l2(value, expected)
        for value, expected in zip(actual_gradients, reference_gradients, strict=True)
    ]
    pytorch_forward_result, triton_forward_result = _measure_pair(
        pytorch_forward,
        triton_forward,
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    pytorch_training_result, triton_training_result = _measure_pair(
        pytorch_forward_backward,
        triton_forward_backward,
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    forward_speedup = cast(
        float, pytorch_forward_result["median_seconds_per_invocation"]
    ) / cast(float, triton_forward_result["median_seconds_per_invocation"])
    training_speedup = cast(
        float, pytorch_training_result["median_seconds_per_invocation"]
    ) / cast(float, triton_training_result["median_seconds_per_invocation"])
    return {
        "shape": {
            "batch": batch,
            "rows": rows,
            "reduction": reduction,
            "columns": columns,
        },
        "dtype": "complex64",
        "layout": "contiguous_batched_matrix",
        "workload_origin": "mps_canonical_transfer_absorption",
        "canonical_sweep_direction": direction,
        "maximum_forward_absolute_error": forward_error,
        "forward_relative_l2_error": _relative_l2(actual, reference),
        "maximum_gradient_absolute_error": max(gradient_errors),
        "gradient_relative_l2_error": max(gradient_relative_errors),
        "pytorch_bmm_forward": pytorch_forward_result,
        "triton_forward": triton_forward_result,
        "pytorch_bmm_forward_backward": pytorch_training_result,
        "triton_forward_backward": triton_training_result,
        "triton_forward_speedup_over_pytorch_bmm": forward_speedup,
        "triton_training_speedup_over_pytorch_bmm": training_speedup,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed NUM-001 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NUM-001 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected = "triton" if args.compiler_lane == "stock_triton" else "flagtree"
    if identity_status != "resolved" or distribution != expected:
        raise RuntimeError(
            f"compiler lane {args.compiler_lane!r} requires {expected!r}; found "
            f"{distribution!r} with status {identity_status!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            *shape,
            seed=args.seed + index,
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
            "ordering": "counterbalanced baseline/candidate by repeat parity",
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
    """Validate a raw run without importing CUDA providers."""

    expected = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    source_revision = payload.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise ValueError("run source_revision must be a full lowercase Git commit SHA")
    for field in ("host_label", "reported_hostname"):
        host_value = payload.get(field)
        if not isinstance(host_value, str) or not host_value.strip():
            raise ValueError(f"run field {field!r} must be a non-empty string")
    lane = payload.get("compiler_lane")
    if lane not in COMPILER_LANES:
        raise ValueError("run records an unsupported compiler lane")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected_distribution = "triton" if lane == "stock_triton" else "flagtree"
    if (
        compiler.get("distribution") != expected_distribution
        or compiler.get("identity_status") != "resolved"
    ):
        raise ValueError("run compiler identity does not match its lane")
    version = compiler.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("run compiler version must be a non-empty string")
    environment = _mapping(payload.get("environment"), "environment")
    gpu = environment.get("gpu")
    if not isinstance(gpu, str) or "A800" not in gpu:
        raise ValueError("run environment must identify an NVIDIA A800 GPU")
    measurement = _mapping(payload.get("measurement"), "measurement")
    warmup = measurement.get("warmup")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("measurement.warmup must be non-negative")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("measurement.repeats must be positive")
    group_size = measurement.get("group_size")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("measurement.group_size must be positive")
    if (
        measurement.get("ordering")
        != "counterbalanced baseline/candidate by repeat parity"
    ):
        raise ValueError("measurement.ordering must be counterbalanced")
    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append(
            (
                *(
                    shape.get(name)
                    for name in ("batch", "rows", "reduction", "columns")
                ),
                case.get("canonical_sweep_direction"),
            )
        )
        if case.get("canonical_sweep_direction") not in (
            "right_going_transfer",
            "left_going_transfer",
        ):
            raise ValueError(f"cases[{index}] records an invalid canonical direction")
        for field, value in (
            ("dtype", "complex64"),
            ("layout", "contiguous_batched_matrix"),
            ("workload_origin", "mps_canonical_transfer_absorption"),
        ):
            if case.get(field) != value:
                raise ValueError(f"cases[{index}].{field} must equal {value!r}")
        for error_field, tolerance in (
            ("maximum_forward_absolute_error", 5e-4),
            ("forward_relative_l2_error", 5e-5),
            ("maximum_gradient_absolute_error", 5e-4),
            ("gradient_relative_l2_error", 5e-5),
        ):
            error = case.get(error_field)
            if not isinstance(error, (int, float)) or not 0 <= error <= tolerance:
                raise ValueError(f"cases[{index}] exceeds {error_field} tolerance")
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds_per_invocation"),
                f"cases[{index}].{result_name}.samples",
            )
            if len(samples) != repeats or any(
                not isinstance(sample, (int, float)) or sample <= 0
                for sample in samples
            ):
                raise ValueError(f"cases[{index}].{result_name} samples are invalid")
            if result.get("median_seconds_per_invocation") != statistics.median(
                samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} median is not reproducible"
                )
            peak_memory = result.get("peak_memory_delta_bytes")
            if not isinstance(peak_memory, int) or peak_memory < 0:
                raise ValueError(f"cases[{index}].{result_name} peak memory is invalid")
        expected_forward = (
            case["pytorch_bmm_forward"]["median_seconds_per_invocation"]
            / case["triton_forward"]["median_seconds_per_invocation"]
        )
        expected_training = (
            case["pytorch_bmm_forward_backward"]["median_seconds_per_invocation"]
            / case["triton_forward_backward"]["median_seconds_per_invocation"]
        )
        if case.get("triton_forward_speedup_over_pytorch_bmm") != expected_forward:
            raise ValueError(f"cases[{index}] forward speedup is not reproducible")
        if case.get("triton_training_speedup_over_pytorch_bmm") != expected_training:
            raise ValueError(f"cases[{index}] training speedup is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed NUM-001 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the two-host, two-compiler matrix into one evidence artifact."""

    if not required_hosts or len(set(required_hosts)) != len(required_hosts):
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload.get("source_revision") for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    observed = {
        (payload.get("host_label"), payload.get("compiler_lane"))
        for payload in payloads
    }
    required = {(host, lane) for host in required_hosts for lane in COMPILER_LANES}
    if observed != required or len(payloads) != len(required):
        raise ValueError("runs must contain every required host/compiler lane once")
    ordered = sorted(
        payloads, key=lambda run: (run["host_label"], run["compiler_lane"])
    )
    forward_wins = all(
        case["triton_forward_speedup_over_pytorch_bmm"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    training_wins = all(
        case["triton_training_speedup_over_pytorch_bmm"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    return {
        "benchmark": "complex_bmm_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "triton_forward_win_on_all_cases": forward_wins,
        "triton_training_win_on_all_cases": training_wins,
        "runtime_dispatch_authorized": forward_wins and training_wins,
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate an aggregate as the canonical merge of its raw runs."""

    hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=hosts)
    if dict(payload) != rebuilt:
        raise ValueError("aggregate is not the canonical merge of its raw runs")


def _read(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--host-label", default=socket.gethostname())
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=271001)
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
    if args.action == "run":
        if args.warmup < 0 or args.repeats <= 0 or args.group_size <= 0:
            raise ValueError("warmup must be non-negative and sample sizes positive")
        payload = collect_run(args)
        validate_run(payload)
        _write(args.output, payload)
    elif args.action == "merge":
        payload = merge_runs(
            [_read(path) for path in args.input], required_hosts=args.required_host
        )
        validate_evidence(payload)
        _write(args.output, payload)
    else:
        validate_evidence(_read(args.input))
        payload = {"validated": str(args.input)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
