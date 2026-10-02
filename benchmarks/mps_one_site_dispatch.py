"""Collect and validate MPS-003 one-site dispatch evidence on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_one_site_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_one_site_dispatch.v1"
SEMANTIC_ID = "mps.contract.one_site_gate"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-003-A"
RUNNER = "benchmarks/mps_one_site_dispatch.py"
SHAPE_MATRIX = (
    (1, 1, 64, 64),
    (4, 2, 32, 32),
    (8, 8, 16, 16),
    (16, 8, 32, 32),
    (32, 8, 64, 64),
)
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_MPS_ONE_SITE"
RESULT_NAMES = (
    "direct_kernel_forward",
    "direct_pytorch_forward",
    "direct_kernel_forward_backward",
    "direct_pytorch_forward_backward",
    "public_catalog_forward",
    "public_pytorch_eager_forward",
    "public_pytorch_compiled_forward",
    "public_catalog_forward_backward",
    "public_pytorch_eager_forward_backward",
    "public_pytorch_compiled_forward_backward",
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
    "public_forward_speedup_over_eager": (
        "public_pytorch_eager_forward",
        "public_catalog_forward",
    ),
    "public_forward_speedup_over_compiled": (
        "public_pytorch_compiled_forward",
        "public_catalog_forward",
    ),
    "public_forward_backward_speedup_over_eager": (
        "public_pytorch_eager_forward_backward",
        "public_catalog_forward_backward",
    ),
    "public_forward_backward_speedup_over_compiled": (
        "public_pytorch_compiled_forward_backward",
        "public_catalog_forward_backward",
    ),
}


def _reference(tensors: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kbpq,kblqr->kblpr", gates, tensors)


def _inputs(
    sites: int,
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = torch.randn(
        sites,
        batch,
        left_dim,
        2,
        right_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    tensors = tensors / torch.linalg.vector_norm(
        tensors, dim=(-3, -2, -1), keepdim=True
    )
    angles = torch.randn(
        sites,
        batch,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    gates = torch.zeros(
        sites,
        batch,
        2,
        2,
        device="cuda",
        dtype=torch.complex64,
    )
    gates[..., 0, 0] = torch.cos(angles / 2)
    gates[..., 0, 1] = -torch.sin(angles / 2)
    gates[..., 1, 0] = torch.sin(angles / 2)
    gates[..., 1, 1] = torch.cos(angles / 2)
    cotangent = torch.randn(
        sites,
        batch,
        left_dim,
        2,
        right_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    return tensors.contiguous(), gates.contiguous(), cotangent.contiguous()


@contextmanager
def _dispatch(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(DISPATCH_VARIABLE)
    os.environ[DISPATCH_VARIABLE] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous


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
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tensors: torch.Tensor,
    gates: torch.Tensor,
    cotangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = function(tensors, gates)
    tensor_gradient, gate_gradient = torch.autograd.grad(
        output,
        (tensors, gates),
        cotangent,
    )
    return output, tensor_gradient, gate_gradient


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
    sites: int,
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.mps_one_site import fused_mps_one_site
    from flagquantum.simulation.mps.site_kernels import (
        apply_ry_bucket,
        reset_site_kernel_stats,
    )

    tensors, gates, cotangent = _inputs(
        sites,
        batch,
        left_dim,
        right_dim,
        seed=seed,
    )
    flat_tensors = tensors.reshape(sites * batch, left_dim, 2, right_dim)
    flat_gates = gates.reshape(sites * batch, 2, 2)
    flat_cotangent = cotangent.reshape(sites * batch, left_dim, 2, right_dim)
    reference = _reference(tensors, gates)
    direct = fused_mps_one_site(flat_tensors, flat_gates).reshape_as(tensors)
    with _dispatch(True):
        public = apply_ry_bucket(tensors, gates, compiled=True)
    with _dispatch(False):
        public_eager = apply_ry_bucket(tensors, gates, compiled=False)
        reset_site_kernel_stats(clear_cache=True)
        public_compiled = apply_ry_bucket(tensors, gates, compiled=True)
    for actual in (direct, public, public_eager, public_compiled):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)

    gradient_tensors = tensors.detach().requires_grad_(True)
    gradient_gates = gates.detach().requires_grad_(True)
    reference_fb = _forward_backward(
        _reference,
        gradient_tensors,
        gradient_gates,
        cotangent,
    )
    direct_fb = _forward_backward(
        lambda tensor, gate: fused_mps_one_site(
            tensor.reshape(sites * batch, left_dim, 2, right_dim),
            gate.reshape(sites * batch, 2, 2),
        ).reshape_as(tensor),
        gradient_tensors,
        gradient_gates,
        cotangent,
    )
    with _dispatch(True):
        public_fb = _forward_backward(
            lambda tensor, gate: apply_ry_bucket(tensor, gate, compiled=True),
            gradient_tensors,
            gradient_gates,
            cotangent,
        )
    for actual, expected in zip(direct_fb, reference_fb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(public_fb, reference_fb, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    direct_kernel_forward = _measure(
        lambda: fused_mps_one_site(flat_tensors, flat_gates),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    direct_pytorch_forward = _measure(
        lambda: torch.einsum("bpq,blqr->blpr", flat_gates, flat_tensors),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )

    direct_gradient_tensors = flat_tensors.detach().requires_grad_(True)
    direct_gradient_gates = flat_gates.detach().requires_grad_(True)
    direct_kernel_forward_backward = _measure(
        lambda: _forward_backward(
            fused_mps_one_site,
            direct_gradient_tensors,
            direct_gradient_gates,
            flat_cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    direct_pytorch_forward_backward = _measure(
        lambda: _forward_backward(
            lambda tensor, gate: torch.einsum("bpq,blqr->blpr", gate, tensor),
            direct_gradient_tensors,
            direct_gradient_gates,
            flat_cotangent,
        ),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )

    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(True):
        public_catalog_forward = _measure(
            lambda: apply_ry_bucket(tensors, gates, compiled=True),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_pytorch_eager_forward = _measure(
            lambda: apply_ry_bucket(tensors, gates, compiled=False),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_pytorch_compiled_forward = _measure(
            lambda: apply_ry_bucket(tensors, gates, compiled=True),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )

    public_gradient_tensors = tensors.detach().requires_grad_(True)
    public_gradient_gates = gates.detach().requires_grad_(True)
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(True):
        public_catalog_forward_backward = _measure(
            lambda: _forward_backward(
                lambda tensor, gate: apply_ry_bucket(tensor, gate, compiled=True),
                public_gradient_tensors,
                public_gradient_gates,
                cotangent,
            ),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_pytorch_eager_forward_backward = _measure(
            lambda: _forward_backward(
                lambda tensor, gate: apply_ry_bucket(tensor, gate, compiled=False),
                public_gradient_tensors,
                public_gradient_gates,
                cotangent,
            ),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_pytorch_compiled_forward_backward = _measure(
            lambda: _forward_backward(
                lambda tensor, gate: apply_ry_bucket(tensor, gate, compiled=True),
                public_gradient_tensors,
                public_gradient_gates,
                cotangent,
            ),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )

    forward_errors = [_error(actual, reference) for actual in (direct, public)]
    gradient_errors = [
        _error(actual, expected)
        for actual_group in (direct_fb[1:], public_fb[1:])
        for actual, expected in zip(actual_group, reference_fb[1:], strict=True)
    ]
    results = {
        "direct_kernel_forward": direct_kernel_forward,
        "direct_pytorch_forward": direct_pytorch_forward,
        "direct_kernel_forward_backward": direct_kernel_forward_backward,
        "direct_pytorch_forward_backward": direct_pytorch_forward_backward,
        "public_catalog_forward": public_catalog_forward,
        "public_pytorch_eager_forward": public_pytorch_eager_forward,
        "public_pytorch_compiled_forward": public_pytorch_compiled_forward,
        "public_catalog_forward_backward": public_catalog_forward_backward,
        "public_pytorch_eager_forward_backward": public_pytorch_eager_forward_backward,
        "public_pytorch_compiled_forward_backward": public_pytorch_compiled_forward_backward,
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
            "sites": sites,
            "batch": batch,
            "left_bond": left_dim,
            "physical_dimension": 2,
            "right_bond": right_dim,
        },
        "contraction_elements": sites * batch * left_dim * right_dim,
        "dtype": "complex64",
        "layout": "contiguous_site_bucket_and_batched_ry_gates",
        "maximum_forward_absolute_error": max(item[0] for item in forward_errors),
        "maximum_forward_relative_l2_error": max(item[1] for item in forward_errors),
        "maximum_gradient_absolute_error": max(item[0] for item in gradient_errors),
        "maximum_gradient_relative_l2_error": max(item[1] for item in gradient_errors),
        **results,
        **ratios,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MPS-003 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-003 benchmark evidence")
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
            sites,
            batch,
            left_dim,
            right_dim,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (sites, batch, left_dim, right_dim) in enumerate(SHAPE_MATRIX)
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
    observed_shapes: list[tuple[int, int, int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shape = (
            shape.get("sites"),
            shape.get("batch"),
            shape.get("left_bond"),
            shape.get("right_bond"),
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
        for error_field, tolerance in (
            ("maximum_forward_absolute_error", 2e-5),
            ("maximum_forward_relative_l2_error", 2e-5),
            ("maximum_gradient_absolute_error", 2e-5),
            ("maximum_gradient_relative_l2_error", 2e-5),
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
        raise ValueError("run does not contain the fixed MPS-003 shape matrix")


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
        "public_forward_win_over_eager_on_all_cases": all_wins(
            "public_forward_speedup_over_eager"
        ),
        "public_forward_win_over_compiled_on_all_cases": all_wins(
            "public_forward_speedup_over_compiled"
        ),
        "public_forward_backward_win_over_eager_on_all_cases": all_wins(
            "public_forward_backward_speedup_over_eager"
        ),
        "public_forward_backward_win_over_compiled_on_all_cases": all_wins(
            "public_forward_backward_speedup_over_compiled"
        ),
    }
    public_wins = all(
        value for key, value in decisions.items() if key.startswith("public_")
    )
    return {
        "benchmark": "mps_one_site_dispatch",
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
        "dispatch_selection_decision": (
            "eligible_for_default" if public_wins else "retain_opt_in"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "mps_one_site_dispatch",
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
    run.add_argument("--seed", type=int, default=270003)
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
