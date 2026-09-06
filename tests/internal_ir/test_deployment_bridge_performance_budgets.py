from __future__ import annotations

import json
import platform
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks.internal import (
    deployment_bridge_stage1_gate,
    deployment_bridge_stage2_gate,
    deployment_bridge_stage3_gate,
    deployment_bridge_stage4_gate,
    deployment_bridge_stage5_gate,
    deployment_bridge_stage6_gate,
)

ROOT = Path(__file__).resolve().parents[2]

STAGES = (
    (
        1,
        deployment_bridge_stage1_gate,
        "inspection_count",
        "deterministic_report_identity",
    ),
    (
        2,
        deployment_bridge_stage2_gate,
        "operation_count",
        "deterministic_evidence_identity",
    ),
    (
        3,
        deployment_bridge_stage3_gate,
        "evaluation_count",
        "deterministic_evidence_identity",
    ),
    (
        4,
        deployment_bridge_stage4_gate,
        "rehearsal_count",
        "deterministic_evidence_identity",
    ),
    (
        5,
        deployment_bridge_stage5_gate,
        "observation_count",
        "deterministic_evidence_identity",
    ),
    (
        6,
        deployment_bridge_stage6_gate,
        "connector_count",
        "deterministic_evidence_identity",
    ),
)


def _budget_platform(stage: int) -> str:
    validation = ROOT / (
        f"contracts/deployment-bridge-stage{stage}-performance-budget-validation.json"
    )
    payload = json.loads(validation.read_text(encoding="utf-8"))
    return payload["environment"]["platform"].partition("-")[0]


@pytest.mark.parametrize(
    ("stage", "gate", "size_key", "identity_key"),
    STAGES,
    ids=[f"stage{stage}" for stage, *_ in STAGES],
)
def test_approved_budget_is_enforced_in_its_qualified_environment(
    stage: int,
    gate: ModuleType,
    size_key: str,
    identity_key: str,
) -> None:
    qualified_platform = _budget_platform(stage)
    if platform.system() != qualified_platform:
        pytest.skip(
            f"stage {stage} latency budget is qualified on {qualified_platform}"
        )

    result = gate.evaluate(
        gate.DEFAULT_BUDGET,
        gate.DEFAULT_AUTHORIZATION,
        iterations=5,
        warmup=2,
    )

    assert result["status"] == "passed", result
    assert [case[size_key] for case in result["cases"]] == [10, 100, 1000, 10000]
    assert all(case["latency_passed"] for case in result["cases"])
    assert all(case["memory_passed"] for case in result["cases"])
    assert all(case[identity_key] for case in result["cases"])


@pytest.mark.parametrize(
    ("stage", "gate"),
    [(stage, gate) for stage, gate, *_ in STAGES],
    ids=[f"stage{stage}" for stage, *_ in STAGES],
)
def test_budget_gate_rejects_an_unbound_budget(
    stage: int,
    gate: ModuleType,
    tmp_path: Path,
) -> None:
    changed_budget = tmp_path / f"stage{stage}-budget.json"
    changed_budget.write_bytes(gate.DEFAULT_BUDGET.read_bytes() + b"\n")

    with pytest.raises(
        ValueError,
        match=f"Stage {stage} budget does not match its authorization",
    ):
        gate.evaluate(
            changed_budget,
            gate.DEFAULT_AUTHORIZATION,
            iterations=5,
            warmup=1,
        )
