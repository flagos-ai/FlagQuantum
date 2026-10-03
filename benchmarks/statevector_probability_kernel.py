"""Collect two-host correctness and performance evidence for MEAS-001."""

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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_probability.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_probability.v1"
SEMANTIC_ID = "measurement.probabilities.statevector"
IMPLEMENTATION_ID = "FQKI-TRITON-MEAS-001-A"
RUNNER = "benchmarks/statevector_probability_kernel.py"
SHAPE_MATRIX = (
    (1, 1 << 10),
    (32, 1 << 10),
    (8, 1 << 16),
    (2, 1 << 20),
    (1, 1 << 24),
)
RESULT_NAMES = (
    "triton_forward",
    "pytorch_forward",
    "triton_forward_backward",
    "pytorch_forward_backward",
)
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _reference(state: torch.Tensor) -> torch.Tensor:
    return torch.abs(state) ** 2


def _input(batch: int, amplitudes: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        batch,
        amplitudes,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    return (state / norm).contiguous()


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
    from flagquantum.kernels.triton.statevector_measurement import (
        statevector_probabilities,
    )

    base = _input(batch, amplitudes, seed=seed)
    weights = torch.randn(
        batch,
        amplitudes,
        generator=torch.Generator(device="cuda").manual_seed(seed + 10_000),
        device="cuda",
        dtype=torch.float32,
    )
    triton_state = base.detach().clone().requires_grad_(True)
    pytorch_state = base.detach().clone().requires_grad_(True)
    triton_output = statevector_probabilities(triton_state)
    pytorch_output = _reference(pytorch_state)
    (triton_gradient,) = torch.autograd.grad(
        triton_output,
        triton_state,
        weights,
    )
    (pytorch_gradient,) = torch.autograd.grad(
        pytorch_output,
        pytorch_state,
        weights,
    )
    torch.testing.assert_close(
        triton_output,
        pytorch_output,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        triton_gradient,
        pytorch_gradient,
        rtol=1e-6,
        atol=1e-7,
    )

    def triton_forward_backward() -> torch.Tensor:
        state = base.detach().requires_grad_(True)
        output = statevector_probabilities(state)
        (gradient,) = torch.autograd.grad(output, state, weights)
        return gradient

    def pytorch_forward_backward() -> torch.Tensor:
        state = base.detach().requires_grad_(True)
        output = _reference(state)
        (gradient,) = torch.autograd.grad(output, state, weights)
        return gradient

    results = {
        "triton_forward": _measure(
            lambda: statevector_probabilities(base),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "pytorch_forward": _measure(
            lambda: _reference(base),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "triton_forward_backward": _measure(
            triton_forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "pytorch_forward_backward": _measure(
            pytorch_forward_backward,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
    }
    probability_difference = triton_output - pytorch_output
    gradient_difference = triton_gradient - pytorch_gradient
    return {
        "shape": {"batch": batch, "amplitudes": amplitudes},
        "dtype": "complex64",
        "layout": "contiguous_flat_statevector",
        "maximum_probability_absolute_error": float(
            torch.max(torch.abs(probability_difference))
        ),
        "probability_relative_l2_error": float(
            torch.linalg.vector_norm(probability_difference)
            / torch.linalg.vector_norm(pytorch_output).clamp_min(
                torch.finfo(torch.float32).eps
            )
        ),
        "maximum_gradient_absolute_error": float(
            torch.max(torch.abs(gradient_difference))
        ),
        "gradient_relative_l2_error": float(
            torch.linalg.vector_norm(gradient_difference)
            / torch.linalg.vector_norm(pytorch_gradient).clamp_min(
                torch.finfo(torch.float32).eps
            )
        ),
        **results,
        "forward_speedup_over_pytorch": (
            float(results["pytorch_forward"]["median_seconds_per_invocation"])
            / float(results["triton_forward"]["median_seconds_per_invocation"])
        ),
        "forward_backward_speedup_over_pytorch": (
            float(results["pytorch_forward_backward"]["median_seconds_per_invocation"])
            / float(results["triton_forward_backward"]["median_seconds_per_invocation"])
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MEAS-001 matrix on one CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MEAS-001 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    if identity_status != "resolved" or distribution != "triton":
        raise RuntimeError(
            "MEAS-001 evidence requires stock Triton, found "
            f"{distribution!r} with status {identity_status!r}"
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
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
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
    if _FULL_REVISION.fullmatch(str(payload.get("source_revision"))) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    for field in ("command", "host_label"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"run field {field!r} must be a non-empty string")
    compiler = _mapping(payload.get("compiler"), "compiler")
    if (
        compiler.get("distribution") != "triton"
        or compiler.get("identity_status") != "resolved"
        or not compiler.get("version")
    ):
        raise ValueError("run must resolve stock Triton compiler identity")
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
    observed_shapes: list[tuple[int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append((shape.get("batch"), shape.get("amplitudes")))
        for error_name in (
            "maximum_probability_absolute_error",
            "probability_relative_l2_error",
            "maximum_gradient_absolute_error",
            "gradient_relative_l2_error",
        ):
            error = case.get(error_name)
            if not isinstance(error, (int, float)) or error > 1e-6:
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
        expected_forward = (
            case["pytorch_forward"]["median_seconds_per_invocation"]
            / case["triton_forward"]["median_seconds_per_invocation"]
        )
        if case.get("forward_speedup_over_pytorch") != expected_forward:
            raise ValueError(f"cases[{index}] forward speedup is not reproducible")
        expected_backward = (
            case["pytorch_forward_backward"]["median_seconds_per_invocation"]
            / case["triton_forward_backward"]["median_seconds_per_invocation"]
        )
        if case.get("forward_backward_speedup_over_pytorch") != expected_backward:
            raise ValueError(
                f"cases[{index}] forward/backward speedup is not reproducible"
            )
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MEAS-001 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    required_hosts: Sequence[str],
) -> dict[str, object]:
    """Combine exactly one stock-Triton run from each required host."""

    if len(set(required_hosts)) != len(required_hosts) or not required_hosts:
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    observed_hosts = [payload["host_label"] for payload in payloads]
    if sorted(observed_hosts) != sorted(required_hosts) or len(payloads) != len(
        required_hosts
    ):
        raise ValueError("runs must contain each required host exactly once")
    ordered = sorted(payloads, key=lambda item: item["host_label"])
    forward_speedups = [
        float(case["forward_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    backward_speedups = [
        float(case["forward_backward_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    return {
        "benchmark": "statevector_probability_kernel",
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
        "compiler_distribution": "triton",
        "forward_speedup_range": [min(forward_speedups), max(forward_speedups)],
        "forward_backward_speedup_range": [
            min(backward_speedups),
            max(backward_speedups),
        ],
        "implementation_decision": "retain_experimental",
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "statevector_probability_kernel",
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
        "compiler_distribution": "triton",
        "implementation_decision": "retain_experimental",
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
    run = subparsers.add_parser("run", help="collect one host run")
    run.add_argument("--host-label", default=socket.gethostname())
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
