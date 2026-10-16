"""Collect two-host public-dispatch evidence for MEAS-001."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_probability_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_probability_dispatch.v1"
SEMANTIC_ID = "measurement.probabilities.statevector"
IMPLEMENTATION_ID = "FQKI-TRITON-MEAS-001-A"
RUNNER = "benchmarks/statevector_probability_dispatch.py"
SHAPE_MATRIX = ((1, 1 << 24),)
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_STATEVECTOR_PROBABILITIES"
RESULT_NAMES = (
    "public_catalog_dispatch_forward",
    "public_pytorch_reference_forward",
    "public_catalog_dispatch_forward_backward",
    "public_pytorch_reference_forward_backward",
)
PERFORMANCE_FLOOR = 1.0
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


@contextmanager
def _dispatch_setting(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(DISPATCH_VARIABLE)
    os.environ[DISPATCH_VARIABLE] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous


def _execute_public(state: torch.Tensor) -> torch.Tensor:
    from flagquantum.runtime.measurements import _joint_marginal_probabilities

    n_wires = int(state.shape[1]).bit_length() - 1
    result = _joint_marginal_probabilities(
        state,
        tuple(range(n_wires)),
        n_wires=n_wires,
        noise_model=None,
    )
    if result is None:
        raise RuntimeError("dense statevector probabilities unexpectedly returned None")
    return result


def _input(batch: int, amplitudes: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        batch,
        amplitudes,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    normalized = (
        state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    ).contiguous()
    if not isinstance(normalized, torch.Tensor):
        raise TypeError("normalization must return a tensor")
    return normalized


def _median(result: Mapping[str, object]) -> float:
    value = result.get("median_seconds_per_invocation")
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark median must be numeric")
    return float(value)


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
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.runtime.statevector_probability_dispatch import (
        _statevector_probability_kernel_enabled,
    )

    base = _input(batch, amplitudes, seed=seed)
    if not _statevector_probability_kernel_enabled(base):
        raise RuntimeError("the fixed MEAS-001 dispatch case did not select the kernel")
    weights = torch.randn(
        batch,
        amplitudes,
        generator=torch.Generator(device="cuda").manual_seed(seed + 10_000),
        device="cuda",
        dtype=torch.float32,
    )

    with _dispatch_setting(True):
        dispatched_state = base.detach().clone().requires_grad_(True)
        dispatched_output = _execute_public(dispatched_state)
        (dispatched_gradient,) = torch.autograd.grad(
            dispatched_output,
            dispatched_state,
            weights,
        )
    with _dispatch_setting(False):
        reference_state = base.detach().clone().requires_grad_(True)
        reference_output = _execute_public(reference_state)
        (reference_gradient,) = torch.autograd.grad(
            reference_output,
            reference_state,
            weights,
        )
    torch.testing.assert_close(
        dispatched_output,
        reference_output,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        dispatched_gradient,
        reference_gradient,
        rtol=1e-6,
        atol=1e-6,
    )

    def forward() -> torch.Tensor:
        return _execute_public(base)

    def forward_backward() -> torch.Tensor:
        state = base.detach().requires_grad_(True)
        output = _execute_public(state)
        return torch.autograd.grad(output, state, weights)[0]

    with _dispatch_setting(True):
        dispatch_forward = _measure(
            forward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
        dispatch_forward_backward = _measure(
            forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    with _dispatch_setting(False):
        reference_forward = _measure(
            forward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
        reference_forward_backward = _measure(
            forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )

    output_difference = dispatched_output - reference_output
    gradient_difference = dispatched_gradient - reference_gradient
    return {
        "shape": {"batch": batch, "amplitudes": amplitudes},
        "dtype": "complex64",
        "layout": "contiguous_flat_statevector",
        "route_expected": True,
        "maximum_probability_absolute_error": float(
            torch.max(torch.abs(output_difference)).detach()
        ),
        "probability_relative_l2_error": float(
            (
                torch.linalg.vector_norm(output_difference)
                / torch.linalg.vector_norm(reference_output).clamp_min(
                    torch.finfo(torch.float32).eps
                )
            ).detach()
        ),
        "maximum_gradient_absolute_error": float(
            torch.max(torch.abs(gradient_difference)).detach()
        ),
        "gradient_relative_l2_error": float(
            (
                torch.linalg.vector_norm(gradient_difference)
                / torch.linalg.vector_norm(reference_gradient).clamp_min(
                    torch.finfo(torch.float32).eps
                )
            ).detach()
        ),
        "public_catalog_dispatch_forward": dispatch_forward,
        "public_pytorch_reference_forward": reference_forward,
        "public_catalog_dispatch_forward_backward": dispatch_forward_backward,
        "public_pytorch_reference_forward_backward": reference_forward_backward,
        "public_forward_speedup_over_pytorch": (
            _median(reference_forward) / _median(dispatch_forward)
        ),
        "public_forward_backward_speedup_over_pytorch": (
            _median(reference_forward_backward) / _median(dispatch_forward_backward)
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the evidenced MEAS-001 public route on one CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MEAS-001 dispatch evidence")
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
            batch,
            amplitudes,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (batch, amplitudes) in enumerate(SHAPE_MATRIX)
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
        "execution_semantics": "single_device_public_runtime_dispatch",
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
        "performance_floor": PERFORMANCE_FLOOR,
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
    """Fail closed when one raw dispatch run misses its contract."""

    expected = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "performance_floor": PERFORMANCE_FLOOR,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    if _FULL_REVISION.fullmatch(str(payload.get("source_revision"))) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    for field in ("command", "host_label"):
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
    observed_matrix: list[tuple[int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        batch = shape.get("batch")
        amplitudes = shape.get("amplitudes")
        if not isinstance(batch, int) or not isinstance(amplitudes, int):
            raise ValueError(f"cases[{index}].shape values must be integers")
        observed_matrix.append((batch, amplitudes))
        if case.get("route_expected") is not True:
            raise ValueError(f"cases[{index}] must exercise the catalog route")
        for error_name in (
            "maximum_probability_absolute_error",
            "probability_relative_l2_error",
            "maximum_gradient_absolute_error",
            "gradient_relative_l2_error",
        ):
            error = case.get(error_name)
            if not isinstance(error, (int, float)) or error > 2e-5:
                raise ValueError(f"cases[{index}].{error_name} exceeds tolerance")
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
        forward_speedup = (
            case["public_pytorch_reference_forward"]["median_seconds_per_invocation"]
            / case["public_catalog_dispatch_forward"]["median_seconds_per_invocation"]
        )
        if case.get("public_forward_speedup_over_pytorch") != forward_speedup:
            raise ValueError(f"cases[{index}] forward speedup is not reproducible")
        backward_speedup = (
            case["public_pytorch_reference_forward_backward"][
                "median_seconds_per_invocation"
            ]
            / case["public_catalog_dispatch_forward_backward"][
                "median_seconds_per_invocation"
            ]
        )
        if case.get("public_forward_backward_speedup_over_pytorch") != backward_speedup:
            raise ValueError(
                f"cases[{index}] forward/backward speedup is not reproducible"
            )
        if forward_speedup < PERFORMANCE_FLOOR:
            raise ValueError(f"cases[{index}] forward speedup misses performance floor")
        if backward_speedup < PERFORMANCE_FLOOR:
            raise ValueError(
                f"cases[{index}] forward/backward speedup misses performance floor"
            )
    if tuple(observed_matrix) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MEAS-001 dispatch matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    required_hosts: Sequence[str],
) -> dict[str, object]:
    """Combine the two-host, two-compiler dispatch matrix."""

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
        payloads,
        key=lambda item: (item["host_label"], item["compiler_lane"]),
    )
    forward_speedups = [
        float(case["public_forward_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    backward_speedups = [
        float(case["public_forward_backward_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    return {
        "benchmark": "statevector_probability_dispatch",
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
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "public_forward_speedup_range": [
            min(forward_speedups),
            max(forward_speedups),
        ],
        "public_forward_backward_speedup_range": [
            min(backward_speedups),
            max(backward_speedups),
        ],
        "public_dispatch_win_on_all_runs": True,
        "dispatch_selection_decision": "eligible_for_default",
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "statevector_probability_dispatch",
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
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "public_dispatch_win_on_all_runs": True,
        "dispatch_selection_decision": "eligible_for_default",
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    run.add_argument("--seed", type=int, default=261105)
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
