from __future__ import annotations

import json

import pytest

from benchmarks.runners.tn.merge_tn_gap_results import merge
from benchmarks.runners.tn.summarize_tn_gap_results import summarize
from benchmarks.runners.tn.tn_gap_common import (
    RESULT_SCHEMA,
    load_workload,
    make_workload,
    replay_pair_path_metrics,
    validate_backend_result,
    write_workload,
)


def _result(backend: str, identity: str, *, status: str = "completed") -> dict:
    return {
        "schema_version": RESULT_SCHEMA,
        "backend": backend,
        "backend_version": "test",
        "workload_identity": identity,
        "comparison_scope": "planning",
        "search_budget_seconds": 1.0,
        "status": status,
        "search_budget_compliant": True,
        "metrics": (
            {
                "search_time_seconds": 0.25 if backend == "flagquantum" else 1.0,
                "estimated_flops": 200 if backend == "flagquantum" else 100,
                "largest_intermediate_elements": (
                    64 if backend == "flagquantum" else 32
                ),
            }
            if status == "completed"
            else {}
        ),
        "distribution_semantics": "planning_only",
        "scalability_claim_allowed": False,
    }


def test_workload_identity_is_stable_and_tamper_evident(tmp_path):
    workload = make_workload(
        name="triangle",
        inputs=((0, 1), (1, 2), (2, 0)),
        shapes=((2, 2), (2, 2), (2, 2)),
        category="small_exact",
    )
    path = tmp_path / "triangle.json"
    write_workload(workload, path)
    assert load_workload(path) == workload

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shapes"][0][0] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent extents|identity"):
        load_workload(path)


def test_result_contract_fails_closed_for_scalability_claim():
    payload = _result("flagquantum", "abc")
    payload["scalability_claim_allowed"] = True
    with pytest.raises(ValueError, match="cannot allow scalability"):
        validate_backend_result(payload)


def test_gap_merge_identifies_largest_planning_gap():
    payload = merge(
        _result("flagquantum", "abc"),
        _result("cotengra", "abc"),
    )
    assert payload["comparable"] is True
    assert payload["largest_gap"] == "path_flops"
    assert payload["ratios"]["estimated_flops_flagquantum_over_cotengra"] == 2.0
    assert "not execution" in payload["claim_boundary"]


def test_gap_merge_reports_search_time_as_a_real_gap():
    flagquantum = _result("flagquantum", "abc")
    cotengra = _result("cotengra", "abc")
    flagquantum["metrics"]["estimated_flops"] = 100
    flagquantum["metrics"]["largest_intermediate_elements"] = 32
    flagquantum["metrics"]["search_time_seconds"] = 2.0
    payload = merge(flagquantum, cotengra)
    assert payload["largest_gap"] == "search_time"
    assert "planning overhead" in payload["next_optimization_target"]


def test_gap_merge_rejects_different_workloads():
    with pytest.raises(ValueError, match="different workloads"):
        merge(
            _result("flagquantum", "abc"),
            _result("cotengra", "xyz"),
        )


def test_gap_summary_retains_all_workloads_and_largest_gap():
    small = merge(
        _result("flagquantum", "small"),
        _result("cotengra", "small"),
    )
    small["workload_name"] = "small"
    large_fq = _result("flagquantum", "large")
    large_ctg = _result("cotengra", "large")
    large_fq["metrics"]["estimated_flops"] = 1000
    large = merge(large_fq, large_ctg)
    large["workload_name"] = "large"
    payload = summarize([small, large], stage="stage_b")
    assert payload["stage"] == "stage_b"
    assert payload["stage_status"] == "completed"
    assert payload["result_count"] == 2
    assert payload["largest_observed_gap"]["workload_name"] == "large"


def test_common_path_replay_reduces_dangling_labels_eagerly():
    workload = make_workload(
        name="dangling_chain",
        inputs=((0, 1), (1, 2), (2,)),
        shapes=((2, 2), (2, 2), (2,)),
        category="small_exact",
    )
    metrics = replay_pair_path_metrics(workload, ((1, 2), (0, 1)))
    assert metrics == {
        "estimated_flops": 8,
        "largest_intermediate_elements": 2,
        "total_intermediate_elements": 3,
    }
