"""Contracts for reproducible MPS-008 public dispatch evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mps_sampling_collapse_dispatch import (
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    MEASUREMENT_ORDERING,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _counterbalanced_orders,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "mps_sampling_collapse_dispatch_a800.json"
)


def _timing(scale: float = 1.0) -> dict[str, object]:
    samples = [scale * value for value in (1e-4, 2e-4, 3e-4, 4e-4)]
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": 4096,
        "peak_memory_delta_bytes": 1024,
    }


def _run(
    host: str,
    lane: str,
    *,
    public_speedup: float = 1.1,
    revision: str = "0123456789abcdef",
) -> dict[str, Any]:
    cases = []
    for batch, right_dim, next_right_dim in SHAPE_MATRIX:
        direct = _timing()
        direct_reference = _timing(1.2)
        public = _timing()
        public_reference = _timing(public_speedup)
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "right_bond": right_dim,
                    "next_right_bond": next_right_dim,
                },
                "dtype": "complex64",
                "layout": "contiguous_mps_sampling_step",
                "maximum_absolute_error": 1e-6,
                "relative_l2_error": 1e-7,
                "direct_kernel_wrapper": direct,
                "direct_pytorch_reference": direct_reference,
                "public_catalog_dispatch": public,
                "public_pytorch_reference": public_reference,
                "direct_kernel_speedup_over_pytorch": 1.2,
                "public_dispatch_speedup_over_pytorch": public_speedup,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": revision,
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": "redacted",
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "3.0.0",
            "integration_path": "direct" if lane == "stock_triton" else "flagtree",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {},
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "ordering": MEASUREMENT_ORDERING,
            "order_cycle_length": len(RESULT_NAMES),
            "warmup": 2,
            "repeats": 4,
            "group_size": 3,
            "seed": 261006,
        },
        "cases": cases,
    }


def _matrix(
    *, public_speedup: float = 1.1, revision: str = "0123456789abcdef"
) -> list[dict[str, Any]]:
    return [
        _run(host, lane, public_speedup=public_speedup, revision=revision)
        for host in HOSTS
        for lane in COMPILER_LANES
    ]


def test_counterbalanced_orders_fill_every_position_once_per_cycle() -> None:
    orders = _counterbalanced_orders(RESULT_NAMES, len(RESULT_NAMES))

    assert len(set(orders)) == len(RESULT_NAMES)
    for position in range(len(RESULT_NAMES)):
        assert {order[position] for order in orders} == set(RESULT_NAMES)


def test_counterbalanced_orders_require_complete_cycles() -> None:
    with pytest.raises(ValueError, match="multiple of operation count"):
        _counterbalanced_orders(RESULT_NAMES, len(RESULT_NAMES) + 1)


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run(HOSTS[0], COMPILER_LANES[0]))


def test_run_validator_recomputes_each_median() -> None:
    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["public_catalog_dispatch"]["median_seconds_per_invocation"] = (
        9.0
    )

    with pytest.raises(ValueError, match="median is not canonical"):
        validate_run(payload)


def test_run_validator_recomputes_public_speedup() -> None:
    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["public_dispatch_speedup_over_pytorch"] = 9.0

    with pytest.raises(ValueError, match="public speedup is not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="matrix is incomplete"):
        merge_runs(_matrix()[:-1])


def test_merge_requires_one_source_revision() -> None:
    matrix = _matrix()
    matrix[-1] = _run(HOSTS[-1], COMPILER_LANES[-1], revision="different")

    with pytest.raises(ValueError, match="one source"):
        merge_runs(matrix)


def test_aggregate_promotes_only_when_public_dispatch_wins_every_case() -> None:
    promoted = merge_runs(_matrix(public_speedup=1.1))
    retained = merge_runs(_matrix(public_speedup=0.99))

    assert promoted["schema"] == EVIDENCE_SCHEMA
    assert promoted["public_dispatch_win_on_all_cases"]
    assert promoted["dispatch_selection_decision"] == "eligible_for_default"
    assert not retained["public_dispatch_win_on_all_cases"]
    assert retained["dispatch_selection_decision"] == "retain_opt_in"


def test_evidence_validator_rejects_noncanonical_summary() -> None:
    payload = merge_runs(_matrix())
    changed = copy.deepcopy(payload)
    changed["dispatch_selection_decision"] = "retain_opt_in"

    with pytest.raises(ValueError, match="canonical merge"):
        validate_evidence(changed)


def test_checked_in_a800_evidence_is_canonical_and_promotes_dispatch() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert payload["source_revision"] == ("c7b30f379ef57576474ab55ead39b96abadce62b")
    assert payload["public_dispatch_win_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"
