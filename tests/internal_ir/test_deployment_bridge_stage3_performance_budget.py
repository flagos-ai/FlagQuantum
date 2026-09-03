from __future__ import annotations

from benchmarks.internal.deployment_bridge_stage3_gate import (
    DEFAULT_AUTHORIZATION,
    DEFAULT_BUDGET,
    evaluate,
)


def test_approved_stage3_budget_is_machine_enforced() -> None:
    result = evaluate(
        DEFAULT_BUDGET,
        DEFAULT_AUTHORIZATION,
        iterations=5,
        warmup=2,
    )

    assert result["status"] == "passed", result
    assert [case["evaluation_count"] for case in result["cases"]] == [
        10,
        100,
        1000,
        10000,
    ]
    assert all(case["latency_passed"] for case in result["cases"])
    assert all(case["memory_passed"] for case in result["cases"])
    assert all(case["deterministic_evidence_identity"] for case in result["cases"])


def test_stage3_gate_rejects_an_unbound_budget(tmp_path) -> None:
    changed_budget = tmp_path / "changed-budget.json"
    changed_budget.write_bytes(DEFAULT_BUDGET.read_bytes() + b"\n")

    try:
        evaluate(
            changed_budget,
            DEFAULT_AUTHORIZATION,
            iterations=5,
            warmup=1,
        )
    except ValueError as error:
        assert str(error) == "Stage 3 budget does not match its authorization"
    else:
        raise AssertionError("unbound Stage 3 budget was accepted")
