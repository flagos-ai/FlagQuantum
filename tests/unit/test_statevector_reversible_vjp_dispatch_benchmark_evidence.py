"""Contracts for reproducible GR-002 dispatch evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.statevector_reversible_vjp_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_ARTIFACT = (
    Path(__file__).parents[2]
    / "benchmarks/results/local/statevector_reversible_vjp_dispatch_a800.json"
)


def _run(host: str, lane: str, *, speedup: float = 1.2) -> dict[str, Any]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, amplitudes, bit_position, default_eligible in SHAPE_MATRIX:
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
                "shape": {
                    "batch": batch,
                    "amplitudes_per_batch": amplitudes,
                    "bit_position": bit_position,
                },
                "dtype": "complex64",
                "layout": "contiguous_flat_statevector",
                "default_eligible": default_eligible,
                "restored_ket_reference_maximum_absolute_value": 2.0,
                "restored_ket_maximum_absolute_error": 0.0,
                "restored_ket_relative_l2_error": 0.0,
                "next_adjoint_reference_maximum_absolute_value": 2.0,
                "next_adjoint_maximum_absolute_error": 0.0,
                "next_adjoint_relative_l2_error": 0.0,
                "gradient_reference": 1.0,
                "gradient_absolute_error": 0.0,
                "gradient_relative_error": 0.0,
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
        "execution_semantics": "single_device_mutating_fast_path",
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
            "group_size": 1,
            "input_restore_in_timed_region": False,
        },
        "cases": cases,
    }


def _matrix(*, speedup: float = 1.2) -> list[dict[str, Any]]:
    return [
        _run(host, lane, speedup=speedup)
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


def test_run_validator_requires_single_mutating_invocation() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["measurement"]["group_size"] = 2
    with pytest.raises(ValueError, match="group_size=1"):
        validate_run(payload)


def test_run_validator_keeps_input_restore_outside_timing() -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["measurement"]["input_restore_in_timed_region"] = True
    with pytest.raises(ValueError, match="outside the timed region"):
        validate_run(payload)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("restored_ket_maximum_absolute_error", "restored_ket tolerance"),
        ("next_adjoint_maximum_absolute_error", "next_adjoint tolerance"),
        ("gradient_absolute_error", "gradient tolerance"),
    ),
)
def test_run_validator_enforces_numerical_tolerances(field: str, message: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = 1.0
    with pytest.raises(ValueError, match=message):
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
    with pytest.raises(ValueError, match=f"misses the {speedup_field} performance floor"):
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
    assert payload["minimum_default_window_speedup_over_pytorch_eager"] == pytest.approx(
        1.3
    )
    assert payload["minimum_default_window_speedup_over_torch_compile"] == pytest.approx(
        1.3
    )
    assert payload["default_window_passed"]
    assert payload["dispatch_decision"] == "eligible_for_default"


def test_checked_in_a800_evidence_is_canonical_and_selects_default() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    validate_evidence(payload)
    assert payload["source_revision"] == (
        "e0638c00870740d9808c64107c3e701252e6e6ce"
    )
    assert payload["required_hosts"] == ["jp-a800-171", "jp-a800-172"]
    assert payload["required_compiler_lanes"] == ["stock_triton", "flagtree"]
    assert payload["minimum_default_window_speedup_over_pytorch_eager"] > 1.0
    assert payload["minimum_default_window_speedup_over_torch_compile"] > 1.0
