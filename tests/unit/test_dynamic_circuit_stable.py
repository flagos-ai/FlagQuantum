from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.dynamic as fqd
import flagquantum.errors as fqe

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "dynamic-circuit-v1.json"
RETIRED_CANDIDATE = ROOT / "contracts" / "dynamic-circuit-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_dynamic_circuit_is_stable_without_stabilizing_execution() -> None:
    contract = _load()

    assert contract["status"] == "stable"
    assert contract["root_manifest_change"] is False
    assert contract["rules"]["construction_api_is_stable_contract"] is True
    assert contract["rules"]["whole_dynamic_api_freeze_implied"] is False
    assert contract["approval"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-10-23",
        "directive": "approve DYNAMIC-CIRCUIT-STABLE-CONTRACT-PROMOTION",
        "approved_scope": "DynamicCircuit construction contract only",
    }
    assert contract["migration"]["candidate_removed"] is True
    assert contract["migration"]["user_code_change_required"] is False
    assert contract["history"]["candidate_owner"] == "API owner"
    assert contract["history"]["candidate_target"] == "first-public-alpha"
    assert (
        "all eight promotion criteria"
        in contract["history"]["candidate_removal_condition"]
    )
    assert not RETIRED_CANDIDATE.exists()
    assert contract["result_boundary"]["native_result_stable"] is False
    # The one signature added during candidacy remains part of the stable history.
    (extension,) = contract["history"]["approved_extensions"]
    assert extension["extension"] == "DynamicCircuit.conditional(any_of=...)"
    assert extension["approved_at"] == "2026-10-20"
    assert "additive keyword-only parameter" in extension["approved_scope"]


def test_dynamic_namespace_matches_stable_contract_exactly() -> None:
    extension = _load()["stable_surface"]

    assert extension["namespace"] == "flagquantum.dynamic"
    assert list(fqd.__all__) == extension["additions"] == ["DynamicCircuit"]
    assert "DynamicCircuit" not in fq.__all__
    assert "DynamicCircuit" not in fq.experimental.dynamic.__all__


def test_dynamic_signatures_match_stable_contract() -> None:
    signatures = _load()["public_signatures"]

    assert (
        str(inspect.signature(fqd.DynamicCircuit, eval_str=False))
        == signatures["DynamicCircuit"]
    )
    for method in ("measure", "reset", "conditional", "state"):
        assert (
            str(inspect.signature(getattr(fqd.DynamicCircuit, method), eval_str=False))
            == signatures[f"DynamicCircuit.{method}"]
        )


def test_dynamic_ir_round_trip_preserves_feedback_semantics() -> None:
    circuit = fqd.DynamicCircuit(2)
    returned = (
        circuit.measure(0, classical_bit=1)
        .reset(0)
        .conditional("x", 1, conditions={1: 1})
    )

    restored = fq.CircuitIR.from_json(circuit.to_ir().to_json())
    restored_builder = fqd.DynamicCircuit.from_ir(restored)
    assert returned is circuit
    assert isinstance(restored_builder, fqd.DynamicCircuit)
    assert restored_builder.to_ir().instructions == restored.instructions
    assert [item.name for item in restored.instructions] == ["measure", "reset", "x"]
    assert restored.instructions[0].metadata["classical_bit"] == 1
    assert restored.instructions[2].metadata["conditions"] == ((1, 1),)


def test_dynamic_builder_uses_stable_error_categories() -> None:
    circuit = fqd.DynamicCircuit(1)
    returned = circuit.x(0)
    assert returned is circuit
    assert isinstance(returned, fqd.DynamicCircuit)
    assert returned.to_ir().instructions[0].name == "x"
    assert returned.state(refresh=True).shape == (1, 2)
    with pytest.raises(fqe.ValidationError):
        fqd.DynamicCircuit(1).measure(-1)
    with pytest.raises(fqe.ValidationError):
        fqd.DynamicCircuit(1).conditional("x", 0, conditions={0: 2})
    with pytest.raises(fqe.CapabilityError):
        fqd.DynamicCircuit(1).measure(0).state()
    with pytest.raises(fqe.CapabilityError):
        (
            fqd.DynamicCircuit(1)
            .measure(0, classical_bit=0)
            .conditional("x", 0, classical_bit=0)
            .state()
        )


def test_disjunctive_condition_survives_the_ir_round_trip() -> None:
    circuit = (
        fqd.DynamicCircuit(2).measure(0, classical_bit=0).measure(1, classical_bit=1)
    )
    circuit.conditional("x", 1, any_of=({0: 1}, {1: 1}))

    restored = fq.CircuitIR.from_json(circuit.to_ir().to_json())

    assert restored.instructions[2].metadata["condition_clauses"] == (
        ((0, 1),),
        ((1, 1),),
    )
    assert "conditions" not in restored.instructions[2].metadata
    rebuilt = fqd.DynamicCircuit.from_ir(restored)
    assert rebuilt.to_ir().instructions == restored.instructions


def test_native_dynamic_result_remains_experimental() -> None:
    assert "DynamicExecutionResult" not in fq.experimental.dynamic.__all__
    assert "DynamicExecutionResult" not in dir(fq.experimental.dynamic)
    with pytest.raises(AttributeError):
        getattr(fq.experimental.dynamic, "DynamicExecutionResult")
    assert "DynamicExecutionResult" not in fqd.__all__
    assert "run_dynamic" not in fqd.__all__
