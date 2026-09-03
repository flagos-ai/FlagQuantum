from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "contracts/ir-phase3-batch-f-performance-baseline.json"
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase3_batch_f_performance_budget_candidate.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_batch_f_performance_candidate_is_unapproved_and_private() -> None:
    baseline = _load(BASELINE)
    candidate = _load(BUDGET)

    assert baseline["status"] == "observed_baseline_not_budget"
    assert candidate["status"] == "candidate_awaiting_owner_approval"
    assert candidate["approval"]["approved"] is False
    assert candidate["approval"]["approval_command"] == (
        "approve IR-PHASE3-BATCH-F-PERFORMANCE-BUDGET"
    )
    assert "not a public SLA" in baseline["claim_scope"]
    assert "not a public SLA" in candidate["claim_scope"]


def test_batch_f_candidate_envelopes_cover_each_observed_baseline() -> None:
    baseline = _load(BASELINE)
    candidate = _load(BUDGET)
    observed = {item["comparison_count"]: item for item in baseline["cases"]}
    budgets = {item["comparison_count"]: item for item in candidate["budgets"]}

    assert set(observed) == set(budgets) == {10, 100, 1000, 10000}
    for count, case in observed.items():
        budget = budgets[count]
        assert case["deterministic_evidence_identity"] is True
        assert case["match_shadow_p95_ms"] <= budget["match_shadow_p95_ms_max"]
        assert (
            case["match_shadow_peak_host_memory_bytes"]
            <= budget["match_shadow_peak_host_memory_bytes_max"]
        )
