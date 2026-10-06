"""Collect and validate MPS-002 projected two-site evidence on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_projected_two_site_dispatch.v2"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_projected_two_site_dispatch.v2"
SEMANTIC_ID = "mps.contract.two_site_gate_projected"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-002-A"
RUNNER = "benchmarks/mps_projected_two_site_dispatch.py"
SHAPE_MATRIX = (
    (1, 16, 16, 16, 8, False),
    (4, 32, 64, 32, 16, False),
    (8, 32, 32, 32, 16, True),
    (8, 64, 32, 64, 32, True),
)
COMPILER_LANES = ("stock_triton", "flagtree")
RESULT_NAMES = (
    "direct_kernel_forward",
    "projected_eager_forward",
    "materialized_pytorch_forward",
    "compiled_projected_reference_forward",
    "catalog_dispatch_forward",
    "public_factorization_forward",
    "eager_factorization_forward",
)
RATIO_BASELINES = {
    "direct_speedup_over_projected_eager": (
        "projected_eager_forward",
        "direct_kernel_forward",
    ),
    "direct_speedup_over_materialized_pytorch": (
        "materialized_pytorch_forward",
        "direct_kernel_forward",
    ),
    "catalog_speedup_over_projected_eager": (
        "projected_eager_forward",
        "catalog_dispatch_forward",
    ),
    "catalog_speedup_over_compiled_projected": (
        "compiled_projected_reference_forward",
        "catalog_dispatch_forward",
    ),
    "public_factorization_speedup_over_eager": (
        "eager_factorization_forward",
        "public_factorization_forward",
    ),
}


def _projected_reference(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Apply the projection before the right-bond contraction."""

    batch, left_dim, _, _ = left.shape
    right_dim = int(right.shape[-1])
    rank = int(projection.shape[-1])
    gate_tensor = (
        gate.reshape(batch, 2, 2, 2, 2) if gate.ndim == 3 else gate.reshape(2, 2, 2, 2)
    )
    projected_right = torch.einsum(
        "bmtr,qrk->bmtqk",
        right,
        projection.reshape(2, right_dim, rank),
    )
    equation = "bpqst,blsm,bmtqk->blpk" if gate.ndim == 3 else "pqst,blsm,bmtqk->blpk"
    return torch.einsum(equation, gate_tensor, left, projected_right).reshape(
        batch, 2 * left_dim, rank
    )


def _materialized_reference(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Form the complete two-site matrix before applying the projection."""

    batch, left_dim, _, _ = left.shape
    right_dim = int(right.shape[-1])
    theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
        batch, left_dim, 4, right_dim
    )
    equation = "bij,bljr->blir" if gate.ndim == 3 else "ij,bljr->blir"
    matrix = torch.einsum(equation, gate, theta).reshape(
        batch, 2 * left_dim, 2 * right_dim
    )
    return matrix @ projection


def _eager_factorization(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the complete fixed-rank route without the catalog kernel."""

    from flagquantum.simulation.mps.low_rank import _cpu_projection

    batch, left_dim, _, _ = left.shape
    right_dim = int(right.shape[-1])
    projection = _cpu_projection(2 * left_dim, 2 * right_dim, rank).to(
        device=left.device,
        dtype=left.dtype,
    )
    sampled = _projected_reference(left, gate, right, projection)
    basis, _ = torch.linalg.qr(sampled, mode="reduced")
    basis_tensor = basis.reshape(batch, left_dim, 2, rank)
    gate_tensor = (
        gate.reshape(batch, 2, 2, 2, 2) if gate.ndim == 3 else gate.reshape(2, 2, 2, 2)
    )
    equation = (
        "blpk,bpqst,blsm,bmtr->bkqr" if gate.ndim == 3 else "blpk,pqst,blsm,bmtr->bkqr"
    )
    reduced = torch.einsum(
        equation,
        torch.conj(basis_tensor),
        gate_tensor,
        left,
        right,
    )
    return basis_tensor, reduced


def _inputs(
    batch: int,
    left_dim: int,
    bond_dim: int,
    right_dim: int,
    rank: int,
    *,
    batched_gate: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from flagquantum.simulation.mps.low_rank import _cpu_projection

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
    gate = torch.zeros((*angle_shape, 4, 4), device="cuda", dtype=torch.complex64)
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
    projection = _cpu_projection(2 * left_dim, 2 * right_dim, rank).to(
        device="cuda",
        dtype=torch.complex64,
    )
    return (
        left.contiguous(),
        gate.contiguous(),
        right.contiguous(),
        projection.contiguous(),
    )


def _measure_counterbalanced(
    functions: Mapping[str, Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, dict[str, object]]:
    names = tuple(functions)
    if not names:
        raise ValueError("at least one benchmark function is required")
    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter()
            for _ in range(group_size):
                functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - started) / group_size)

    results: dict[str, dict[str, object]] = {}
    for name, function in functions.items():
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        function()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        results[name] = {
            "samples_seconds_per_invocation": samples[name],
            "median_seconds_per_invocation": statistics.median(samples[name]),
            "peak_memory_bytes": peak,
            "peak_memory_delta_bytes": max(0, peak - allocated_before),
        }
    return results


def _error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual - expected
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return (
        float(torch.max(torch.abs(difference))),
        float(torch.linalg.vector_norm(difference) / denominator),
    )


def _factorization_errors(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[float, float], tuple[float, float]]:
    actual_basis, actual_reduced = actual
    expected_basis, expected_reduced = expected
    batch = int(actual_basis.shape[0])
    rank = int(actual_basis.shape[-1])
    actual_matrix = actual_basis.reshape(batch, -1, rank) @ actual_reduced.reshape(
        batch, rank, -1
    )
    expected_matrix = expected_basis.reshape(
        batch, -1, rank
    ) @ expected_reduced.reshape(batch, rank, -1)
    actual_flat_basis = actual_basis.reshape(batch, -1, rank)
    expected_flat_basis = expected_basis.reshape(batch, -1, rank)
    actual_projector = actual_flat_basis @ torch.conj(actual_flat_basis).transpose(
        -2, -1
    )
    expected_projector = expected_flat_basis @ torch.conj(
        expected_flat_basis
    ).transpose(-2, -1)
    return _error(actual_matrix, expected_matrix), _error(
        actual_projector, expected_projector
    )


def _case(
    batch: int,
    left_dim: int,
    bond_dim: int,
    right_dim: int,
    rank: int,
    *,
    batched_gate: bool,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.mps_two_site import fused_mps_range_projection
    from flagquantum.simulation.mps.low_rank import fixed_rank_two_site_range_qr
    from flagquantum.simulation.mps.two_site_dispatch import (
        _apply_cataloged_mps_projected_two_site,
    )

    left, gate, right, projection = _inputs(
        batch,
        left_dim,
        bond_dim,
        right_dim,
        rank,
        batched_gate=batched_gate,
        seed=seed,
    )
    compiled_reference = torch.compile(_projected_reference, fullgraph=True)
    projected_reference = _projected_reference(left, gate, right, projection)
    materialized_reference = _materialized_reference(left, gate, right, projection)
    direct = fused_mps_range_projection(left, gate, right, projection)
    catalog = _apply_cataloged_mps_projected_two_site(left, gate, right, projection)
    compiled = compiled_reference(left, gate, right, projection)
    for actual in (materialized_reference, direct, catalog, compiled):
        torch.testing.assert_close(
            actual,
            projected_reference,
            rtol=5e-4,
            atol=5e-4,
        )

    public_factorization = fixed_rank_two_site_range_qr(left, gate, right, rank)
    eager_factorization = _eager_factorization(left, gate, right, rank)
    reconstruction_error, subspace_error = _factorization_errors(
        public_factorization,
        eager_factorization,
    )
    if max(*reconstruction_error, *subspace_error) > 2e-3:
        raise AssertionError("public and eager fixed-rank factorizations diverged")

    results = _measure_counterbalanced(
        {
            "direct_kernel_forward": lambda: fused_mps_range_projection(
                left, gate, right, projection
            ),
            "projected_eager_forward": lambda: _projected_reference(
                left, gate, right, projection
            ),
            "materialized_pytorch_forward": lambda: _materialized_reference(
                left, gate, right, projection
            ),
            "compiled_projected_reference_forward": lambda: compiled_reference(
                left, gate, right, projection
            ),
            "catalog_dispatch_forward": lambda: _apply_cataloged_mps_projected_two_site(
                left, gate, right, projection
            ),
            "public_factorization_forward": lambda: fixed_rank_two_site_range_qr(
                left, gate, right, rank
            ),
            "eager_factorization_forward": lambda: _eager_factorization(
                left, gate, right, rank
            ),
        },
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    ratios = {
        field: (
            float(results[baseline]["median_seconds_per_invocation"])
            / float(results[candidate]["median_seconds_per_invocation"])
        )
        for field, (baseline, candidate) in RATIO_BASELINES.items()
    }
    sample_errors = [
        _error(actual, projected_reference)
        for actual in (materialized_reference, direct, catalog, compiled)
    ]
    full_matrix_elements = batch * (2 * left_dim) * (2 * right_dim)
    projected_output_elements = batch * (2 * left_dim) * rank
    return {
        "shape": {
            "batch": batch,
            "left_bond": left_dim,
            "physical_dimension": 2,
            "middle_bond": bond_dim,
            "right_bond": right_dim,
            "rank": rank,
            "batched_gate": batched_gate,
        },
        "contraction_elements": batch * left_dim * bond_dim * right_dim * rank,
        "full_matrix_elements": full_matrix_elements,
        "projected_output_elements": projected_output_elements,
        "avoided_materialization_ratio": (
            full_matrix_elements / projected_output_elements
        ),
        "runtime_inference_eligible": True,
        "dtype": "complex64",
        "layout": "contiguous_mps_two_site_projected_range",
        "maximum_sample_absolute_error": max(item[0] for item in sample_errors),
        "maximum_sample_relative_l2_error": max(item[1] for item in sample_errors),
        "factorization_reconstruction_absolute_error": reconstruction_error[0],
        "factorization_reconstruction_relative_l2_error": reconstruction_error[1],
        "factorization_subspace_absolute_error": subspace_error[0],
        "factorization_subspace_relative_l2_error": subspace_error[1],
        **results,
        **ratios,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MPS-002 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-002 benchmark evidence")
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
    os.environ["FQ_TRITON_MPS_PROJECTED_TWO_SITE"] = "1"
    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            batch,
            left_dim,
            bond_dim,
            right_dim,
            rank,
            batched_gate=batched_gate,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (
            batch,
            left_dim,
            bond_dim,
            right_dim,
            rank,
            batched_gate,
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
        "execution_semantics": "single_device_fast_path",
        "rollout_environment": {
            "FQ_TRITON_MPS_PROJECTED_TWO_SITE": "1",
        },
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
            "timing_order": "counterbalanced_forward_reverse_per_repeat",
            "memory_collection": "separate_single_invocation_after_timing",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": args.group_size,
            "seed": args.seed,
            "directions": ["forward"],
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
        "rollout_environment": {
            "FQ_TRITON_MPS_PROJECTED_TWO_SITE": "1",
        },
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
    if measurement.get("directions") != ["forward"]:
        raise ValueError("measurement directions must cover forward only")
    if measurement.get("timing_order") != "counterbalanced_forward_reverse_per_repeat":
        raise ValueError("measurement timing_order must be counterbalanced")
    if (
        measurement.get("memory_collection")
        != "separate_single_invocation_after_timing"
    ):
        raise ValueError("measurement memory_collection must be separate from timing")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[int, int, int, int, int, bool]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shape = (
            shape.get("batch"),
            shape.get("left_bond"),
            shape.get("middle_bond"),
            shape.get("right_bond"),
            shape.get("rank"),
            shape.get("batched_gate"),
        )
        observed_shapes.append(observed_shape)
        batch, left_dim, bond_dim, right_dim, rank, _ = observed_shape
        expected_contraction = batch * left_dim * bond_dim * right_dim * rank
        expected_full = batch * (2 * left_dim) * (2 * right_dim)
        expected_projected = batch * (2 * left_dim) * rank
        if case.get("contraction_elements") != expected_contraction:
            raise ValueError(f"cases[{index}] contraction_elements is not reproducible")
        if case.get("full_matrix_elements") != expected_full:
            raise ValueError(f"cases[{index}] full_matrix_elements is not reproducible")
        if case.get("projected_output_elements") != expected_projected:
            raise ValueError(
                f"cases[{index}] projected_output_elements is not reproducible"
            )
        if case.get("avoided_materialization_ratio") != (
            expected_full / expected_projected
        ):
            raise ValueError(
                f"cases[{index}] avoided_materialization_ratio is not reproducible"
            )
        if case.get("runtime_inference_eligible") is not True:
            raise ValueError(f"cases[{index}] runtime_inference_eligible must be true")
        for error_field in (
            "maximum_sample_absolute_error",
            "maximum_sample_relative_l2_error",
            "factorization_reconstruction_absolute_error",
            "factorization_reconstruction_relative_l2_error",
            "factorization_subspace_absolute_error",
            "factorization_subspace_relative_l2_error",
        ):
            error = case.get(error_field)
            if not isinstance(error, (int, float)) or error > 2e-3:
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
        raise ValueError("run does not contain the fixed MPS-002 shape matrix")


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
        payloads,
        key=lambda item: (item["host_label"], item["compiler_lane"]),
    )

    def all_wins(field: str) -> bool:
        return all(case[field] > 1.0 for run in ordered for case in run["cases"])

    decisions = {
        "direct_win_over_projected_eager_on_all_cases": all_wins(
            "direct_speedup_over_projected_eager"
        ),
        "direct_win_over_materialized_pytorch_on_all_cases": all_wins(
            "direct_speedup_over_materialized_pytorch"
        ),
        "catalog_win_over_projected_eager_on_all_cases": all_wins(
            "catalog_speedup_over_projected_eager"
        ),
        "catalog_win_over_compiled_projected_on_all_cases": all_wins(
            "catalog_speedup_over_compiled_projected"
        ),
        "public_factorization_win_over_eager_on_all_cases": all_wins(
            "public_factorization_speedup_over_eager"
        ),
        "direct_memory_win_over_projected_eager_on_all_cases": all(
            case["direct_kernel_forward"]["peak_memory_delta_bytes"]
            < case["projected_eager_forward"]["peak_memory_delta_bytes"]
            for run in ordered
            for case in run["cases"]
        ),
        "direct_memory_win_over_materialized_pytorch_on_all_cases": all(
            case["direct_kernel_forward"]["peak_memory_delta_bytes"]
            < case["materialized_pytorch_forward"]["peak_memory_delta_bytes"]
            for run in ordered
            for case in run["cases"]
        ),
    }
    eligible_for_default = (
        decisions["catalog_win_over_projected_eager_on_all_cases"]
        and decisions["public_factorization_win_over_eager_on_all_cases"]
    )
    return {
        "benchmark": "mps_projected_two_site_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "rollout_environment": {
            "FQ_TRITON_MPS_PROJECTED_TWO_SITE": "1",
        },
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
        "projected_kernel_dispatch_decision": (
            "eligible_for_default" if eligible_for_default else "retain_opt_in"
        ),
        "projected_kernel_dispatch_blockers": (
            []
            if eligible_for_default
            else [
                "projected kernel does not beat eager sampling and end-to-end "
                "factorization on every evidenced shape"
            ]
        ),
        "fixed_rank_rollout_decision": "retain_opt_in",
        "fixed_rank_rollout_blockers": [
            "fixed-rank QR is approximate and does not measure discarded weight"
        ],
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "mps_projected_two_site_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "rollout_environment": {
            "FQ_TRITON_MPS_PROJECTED_TWO_SITE": "1",
        },
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
        "fixed_rank_rollout_decision": "retain_opt_in",
        "fixed_rank_rollout_blockers": [
            "fixed-rank QR is approximate and does not measure discarded weight"
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
    run.add_argument("--seed", type=int, default=270002)
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
