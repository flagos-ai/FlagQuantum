"""Collect and validate MPS-001 two-site dispatch evidence on one CUDA device."""

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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_two_site_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_two_site_dispatch.v1"
SEMANTIC_ID = "mps.contract.two_site_gate"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-001-A"
RUNNER = "benchmarks/mps_two_site_dispatch.py"
SHAPE_MATRIX = (
    (1, 16, 16, 16, False),
    (4, 32, 64, 32, False),
    (8, 32, 32, 32, True),
    (8, 64, 32, 64, True),
)
COMPILER_LANES = ("stock_triton", "flagtree")
RESULT_NAMES = (
    "direct_kernel_forward",
    "direct_pytorch_forward",
    "direct_kernel_forward_backward",
    "direct_pytorch_forward_backward",
    "catalog_dispatch_forward",
    "eager_reference_forward",
    "compiled_reference_forward",
    "catalog_dispatch_forward_backward",
    "eager_reference_forward_backward",
    "compiled_reference_forward_backward",
)
RATIO_BASELINES = {
    "direct_forward_speedup_over_pytorch": (
        "direct_pytorch_forward",
        "direct_kernel_forward",
    ),
    "direct_forward_backward_speedup_over_pytorch": (
        "direct_pytorch_forward_backward",
        "direct_kernel_forward_backward",
    ),
    "catalog_forward_speedup_over_eager": (
        "eager_reference_forward",
        "catalog_dispatch_forward",
    ),
    "catalog_forward_speedup_over_compiled": (
        "compiled_reference_forward",
        "catalog_dispatch_forward",
    ),
    "catalog_forward_backward_speedup_over_eager": (
        "eager_reference_forward_backward",
        "catalog_dispatch_forward_backward",
    ),
    "catalog_forward_backward_speedup_over_compiled": (
        "compiled_reference_forward_backward",
        "catalog_dispatch_forward_backward",
    ),
}


def _reference(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
        left.shape[0], left.shape[1], 4, right.shape[3]
    )
    equation = "bij,bljr->blir" if gate.ndim == 3 else "ij,bljr->blir"
    return torch.einsum(equation, gate, theta).reshape(
        left.shape[0], left.shape[1] * 2, 2 * right.shape[3]
    )


def _inputs(
    batch: int,
    left_dim: int,
    bond_dim: int,
    right_dim: int,
    *,
    batched_gate: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    left = torch.randn(
        batch,
        left_dim,
        2,
        bond_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    right = torch.randn(
        batch,
        bond_dim,
        2,
        right_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    left = left / torch.linalg.vector_norm(left, dim=(-3, -2, -1), keepdim=True)
    right = right / torch.linalg.vector_norm(right, dim=(-3, -2, -1), keepdim=True)
    angle_shape = (batch,) if batched_gate else (1,)
    angles = torch.randn(
        angle_shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    gate = torch.zeros(
        (*angle_shape, 4, 4),
        device="cuda",
        dtype=torch.complex64,
    )
    cosine = torch.cos(angles / 2).to(torch.complex64)
    sine = -1j * torch.sin(angles / 2)
    diagonal = torch.arange(4, device="cuda")
    gate[..., diagonal, diagonal] = cosine[..., None]
    gate[..., 0, 3] = sine
    gate[..., 1, 2] = sine
    gate[..., 2, 1] = sine
    gate[..., 3, 0] = sine
    if not batched_gate:
        gate = gate[0]
    cotangent = torch.randn(
        batch,
        left_dim * 2,
        2 * right_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    return (
        left.contiguous(),
        gate.contiguous(),
        right.contiguous(),
        cotangent.contiguous(),
    )


def _measure(
    function: Callable[[], object],
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
    peak = torch.cuda.max_memory_allocated()
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": peak,
        "peak_memory_delta_bytes": max(0, peak - allocated_before),
    }


def _forward_backward(
    function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    cotangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = function(left, gate, right)
    left_gradient, gate_gradient, right_gradient = torch.autograd.grad(
        output,
        (left, gate, right),
        cotangent,
    )
    return output, left_gradient, gate_gradient, right_gradient


def _error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    difference = actual - expected
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return (
        float(torch.max(torch.abs(difference))),
        float(torch.linalg.vector_norm(difference) / denominator),
    )


def _case(
    batch: int,
    left_dim: int,
    bond_dim: int,
    right_dim: int,
    *,
    batched_gate: bool,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.mps_two_site import fused_mps_two_site
    from flagquantum.simulation.mps.two_site_dispatch import (
        _apply_cataloged_mps_two_site,
    )

    left, gate, right, cotangent = _inputs(
        batch,
        left_dim,
        bond_dim,
        right_dim,
        batched_gate=batched_gate,
        seed=seed,
    )
    compiled_reference = torch.compile(_reference, fullgraph=True)
    reference = _reference(left, gate, right)
    direct = fused_mps_two_site(left, gate, right)
    catalog = _apply_cataloged_mps_two_site(left, gate, right)
    eager_reference = _reference(left, gate, right)
    compiled_reference_output = compiled_reference(left, gate, right)
    for actual in (direct, catalog, eager_reference, compiled_reference_output):
        torch.testing.assert_close(actual, reference, rtol=5e-5, atol=5e-5)

    gradient_left = left.detach().requires_grad_(True)
    gradient_gate = gate.detach().requires_grad_(True)
    gradient_right = right.detach().requires_grad_(True)
    reference_fb = _forward_backward(
        _reference,
        gradient_left,
        gradient_gate,
        gradient_right,
        cotangent,
    )
    direct_fb = _forward_backward(
        fused_mps_two_site,
        gradient_left,
        gradient_gate,
        gradient_right,
        cotangent,
    )
    catalog_fb = _forward_backward(
        _apply_cataloged_mps_two_site,
        gradient_left,
        gradient_gate,
        gradient_right,
        cotangent,
    )
    compiled_fb = _forward_backward(
        compiled_reference,
        gradient_left,
        gradient_gate,
        gradient_right,
        cotangent,
    )
    for actual, expected in zip(direct_fb, reference_fb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for actual, expected in zip(catalog_fb, reference_fb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for actual, expected in zip(compiled_fb, reference_fb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)

    direct_kernel_forward = _measure(
        lambda: fused_mps_two_site(left, gate, right),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    direct_pytorch_forward = _measure(
        lambda: _reference(left, gate, right),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )

    direct_kernel_forward_backward = _measure(
        lambda: _forward_backward(
            fused_mps_two_site,
            gradient_left,
            gradient_gate,
            gradient_right,
            cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    direct_pytorch_forward_backward = _measure(
        lambda: _forward_backward(
            _reference,
            gradient_left,
            gradient_gate,
            gradient_right,
            cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )

    catalog_dispatch_forward = _measure(
        lambda: _apply_cataloged_mps_two_site(left, gate, right),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    eager_reference_forward = _measure(
        lambda: _reference(left, gate, right),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    compiled_reference_forward = _measure(
        lambda: compiled_reference(left, gate, right),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    catalog_dispatch_forward_backward = _measure(
        lambda: _forward_backward(
            _apply_cataloged_mps_two_site,
            gradient_left,
            gradient_gate,
            gradient_right,
            cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    eager_reference_forward_backward = _measure(
        lambda: _forward_backward(
            _reference,
            gradient_left,
            gradient_gate,
            gradient_right,
            cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    compiled_reference_forward_backward = _measure(
        lambda: _forward_backward(
            compiled_reference,
            gradient_left,
            gradient_gate,
            gradient_right,
            cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )

    forward_errors = [_error(actual, reference) for actual in (direct, catalog)]
    gradient_errors = [
        _error(actual, expected)
        for actual_group in (direct_fb[1:], catalog_fb[1:], compiled_fb[1:])
        for actual, expected in zip(actual_group, reference_fb[1:], strict=True)
    ]
    results = {
        "direct_kernel_forward": direct_kernel_forward,
        "direct_pytorch_forward": direct_pytorch_forward,
        "direct_kernel_forward_backward": direct_kernel_forward_backward,
        "direct_pytorch_forward_backward": direct_pytorch_forward_backward,
        "catalog_dispatch_forward": catalog_dispatch_forward,
        "eager_reference_forward": eager_reference_forward,
        "compiled_reference_forward": compiled_reference_forward,
        "catalog_dispatch_forward_backward": catalog_dispatch_forward_backward,
        "eager_reference_forward_backward": eager_reference_forward_backward,
        "compiled_reference_forward_backward": compiled_reference_forward_backward,
    }
    ratios = {
        field: (
            float(results[baseline]["median_seconds_per_invocation"])
            / float(results[candidate]["median_seconds_per_invocation"])
        )
        for field, (baseline, candidate) in RATIO_BASELINES.items()
    }
    return {
        "shape": {
            "batch": batch,
            "left_bond": left_dim,
            "physical_dimension": 2,
            "middle_bond": bond_dim,
            "right_bond": right_dim,
            "batched_gate": batched_gate,
        },
        "contraction_elements": batch * left_dim * bond_dim * right_dim,
        "runtime_forward_eligible": (batch * left_dim * bond_dim * right_dim >= 2**12),
        "runtime_backward_eligible": (batch * left_dim * bond_dim * right_dim >= 2**18),
        "dtype": "complex64",
        "layout": "contiguous_mps_two_site_and_shared_or_batched_gate",
        "maximum_forward_absolute_error": max(item[0] for item in forward_errors),
        "maximum_forward_relative_l2_error": max(item[1] for item in forward_errors),
        "maximum_gradient_absolute_error": max(item[0] for item in gradient_errors),
        "maximum_gradient_relative_l2_error": max(item[1] for item in gradient_errors),
        **results,
        **ratios,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MPS-001 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-001 benchmark evidence")
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
            left_dim,
            bond_dim,
            right_dim,
            batched_gate=batched_gate,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (batch, left_dim, bond_dim, right_dim, batched_gate) in enumerate(
            SHAPE_MATRIX
        )
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
            "directions": ["forward", "forward_backward"],
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
    if measurement.get("directions") != ["forward", "forward_backward"]:
        raise ValueError("measurement directions must cover forward and backward")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[int, int, int, int, bool]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shape = (
            shape.get("batch"),
            shape.get("left_bond"),
            shape.get("middle_bond"),
            shape.get("right_bond"),
            shape.get("batched_gate"),
        )
        observed_shapes.append(observed_shape)
        expected_elements = (
            observed_shape[0]
            * observed_shape[1]
            * observed_shape[2]
            * observed_shape[3]
        )
        if case.get("contraction_elements") != expected_elements:
            raise ValueError(f"cases[{index}] contraction_elements is not reproducible")
        if case.get("runtime_forward_eligible") != (expected_elements >= 2**12):
            raise ValueError(
                f"cases[{index}] runtime_forward_eligible is not reproducible"
            )
        if case.get("runtime_backward_eligible") != (expected_elements >= 2**18):
            raise ValueError(
                f"cases[{index}] runtime_backward_eligible is not reproducible"
            )
        for error_field, tolerance in (
            ("maximum_forward_absolute_error", 5e-5),
            ("maximum_forward_relative_l2_error", 5e-5),
            ("maximum_gradient_absolute_error", 5e-5),
            ("maximum_gradient_relative_l2_error", 5e-5),
        ):
            error = case.get(error_field)
            if not isinstance(error, (int, float)) or error > tolerance:
                raise ValueError(f"cases[{index}] exceeds {error_field} tolerance")
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
        for field, (baseline, candidate) in RATIO_BASELINES.items():
            ratio = (
                case[baseline]["median_seconds_per_invocation"]
                / case[candidate]["median_seconds_per_invocation"]
            )
            if case.get(field) != ratio:
                raise ValueError(f"cases[{index}] {field} is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MPS-001 shape matrix")


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

    def all_wins(field: str) -> bool:
        return all(case[field] > 1.0 for run in ordered for case in run["cases"])

    decisions = {
        "direct_forward_win_on_all_cases": all_wins(
            "direct_forward_speedup_over_pytorch"
        ),
        "direct_forward_backward_win_on_all_cases": all_wins(
            "direct_forward_backward_speedup_over_pytorch"
        ),
        "catalog_forward_win_over_eager_on_all_cases": all_wins(
            "catalog_forward_speedup_over_eager"
        ),
        "catalog_forward_win_over_compiled_on_all_cases": all_wins(
            "catalog_forward_speedup_over_compiled"
        ),
        "catalog_forward_backward_win_over_eager_on_all_cases": all_wins(
            "catalog_forward_backward_speedup_over_eager"
        ),
        "catalog_forward_backward_win_over_compiled_on_all_cases": all_wins(
            "catalog_forward_backward_speedup_over_compiled"
        ),
    }
    return {
        "benchmark": "mps_two_site_dispatch",
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
        **decisions,
        "dispatch_selection_decision": "retain_opt_in",
        "default_dispatch_blockers": [
            "catalog-route evidence excludes the downstream MPS factorization"
        ],
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "mps_two_site_dispatch",
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
        "dispatch_selection_decision": "retain_opt_in",
        "default_dispatch_blockers": [
            "catalog-route evidence excludes the downstream MPS factorization"
        ],
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
    run.add_argument("--seed", type=int, default=270001)
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
