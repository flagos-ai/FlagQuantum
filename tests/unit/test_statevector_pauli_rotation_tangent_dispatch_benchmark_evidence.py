"""Contracts for reproducible GR-005 dispatch evidence."""

from __future__ import annotations

import copy
import statistics
from typing import Any

import pytest
import torch

from benchmarks.statevector_pauli_rotation_tangent_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _pytorch_reference,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _run(host: str, lane: str, *, speedup: float = 1.2) -> dict[str, Any]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, pairs, depth, default_eligible in SHAPE_MATRIX:
        dispatch = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
        }
        baseline_samples = [sample * speedup for sample in samples]
        baseline = {
            "samples_seconds_per_invocation": baseline_samples,
            "median_seconds_per_invocation": statistics.median(baseline_samples),
        }
        cases.append(
            {
                "shape": {"batch": batch, "pairs": pairs, "depth": depth},
                "dtype": "complex64",
                "layout": "parameter_major_tangents",
                "default_eligible": default_eligible,
                "tangent_reference_maximum_absolute_value": 2.0,
                "tangent_maximum_absolute_error": 0.0,
                "tangent_relative_l2_error": 0.0,
                "catalog_dispatch": copy.deepcopy(dispatch),
                "pytorch_eager": copy.deepcopy(baseline),
                "torch_compile": copy.deepcopy(baseline),
                "speedup_over_pytorch_eager": speedup,
                "speedup_over_torch_compile": speedup,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    integration = "direct" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": _REVISION,
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "compiler_lane": lane,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": distribution,
            "version": "3.7.1",
            "integration_path": integration,
            "backend": "cuda",
            "identity_status": "resolved",
        },
        "measurement": {
            "warmup": 10,
            "repeats": len(samples),
            "group_size": 10,
        },
        "cases": cases,
    }


def _matrix(*, speedup: float = 1.2) -> list[dict[str, Any]]:
    return [
        _run(host, lane, speedup=speedup)
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def _forward(state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    output = state
    for layer in range(int(angles.shape[1])):
        for family in range(3):
            zero, one, two, three = output.unbind(dim=-1)
            if family == 0:
                transformed = torch.stack((three, two, one, zero), dim=-1)
            elif family == 1:
                transformed = torch.stack((-three, two, one, -zero), dim=-1)
            else:
                transformed = torch.stack((zero, -one, -two, three), dim=-1)
            half = 0.5 * angles[:, layer, family].reshape(-1, 1)
            output = torch.cos(half) * output - 1j * torch.sin(half) * transformed
    return output


def test_analytic_tangent_reference_matches_finite_differences() -> None:
    generator = torch.Generator().manual_seed(41)
    state = torch.randn(1, 2, 4, dtype=torch.complex128, generator=generator)
    angles = torch.randn(1, 2, 3, dtype=torch.float64, generator=generator)
    epsilon = 1e-6
    finite_differences = []
    for layer in range(2):
        for family in range(3):
            perturbation = torch.zeros_like(angles)
            perturbation[:, layer, family] = epsilon
            finite_differences.append(
                (
                    _forward(state, angles + perturbation)
                    - _forward(state, angles - perturbation)
                )
                / (2 * epsilon)
            )
    torch.testing.assert_close(
        _pytorch_reference(state, angles),
        torch.stack(finite_differences),
        atol=2e-9,
        rtol=2e-7,
    )


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


def test_run_validator_recomputes_medians() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["catalog_dispatch"]["median_seconds_per_invocation"] = 9.0
    with pytest.raises(ValueError, match="median is invalid"):
        validate_run(payload)


def test_run_validator_enforces_numerical_tolerances() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0]["tangent_maximum_absolute_error"] = 1.0
    with pytest.raises(ValueError, match="tangent tolerance"):
        validate_run(payload)


@pytest.mark.parametrize(
    "speedup_field,baseline",
    (
        ("speedup_over_pytorch_eager", "pytorch_eager"),
        ("speedup_over_torch_compile", "torch_compile"),
    ),
)
def test_run_validator_enforces_each_default_window_floor(
    speedup_field: str, baseline: str
) -> None:
    payload = _run("jp-a800-171", "stock_triton", speedup=1.0)
    validate_run(payload)
    payload["cases"][0][baseline]["samples_seconds_per_invocation"] = [
        0.99e-4,
        1.98e-4,
        2.97e-4,
    ]
    payload["cases"][0][baseline]["median_seconds_per_invocation"] = 1.98e-4
    payload["cases"][0][speedup_field] = 1.98e-4 / 2e-4
    with pytest.raises(
        ValueError, match=f"misses the {speedup_field} performance floor"
    ):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="every host/compiler lane"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_selects_the_default_window() -> None:
    payload = merge_runs(
        _matrix(speedup=1.3),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )
    validate_evidence(payload)
    assert payload[
        "minimum_default_window_speedup_over_pytorch_eager"
    ] == pytest.approx(1.3)
    assert payload[
        "minimum_default_window_speedup_over_torch_compile"
    ] == pytest.approx(1.3)
    assert payload["default_window_passed"]
    assert payload["dispatch_decision"] == "eligible_for_default"
