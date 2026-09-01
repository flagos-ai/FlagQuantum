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
CONTRACT = ROOT / "contracts" / "dynamic-circuit-v1-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_dynamic_circuit_is_candidate_stable_not_frozen() -> None:
    candidate = _load()

    assert candidate["status"] == "implemented_pending_review"
    assert candidate["implementation_authorized"] is True
    assert candidate["root_manifest_change"] is False
    assert candidate["rules"]["candidate_is_frozen_contract"] is False
    assert candidate["result_boundary"]["native_result_candidate_stable"] is False


def test_dynamic_namespace_matches_candidate_exactly() -> None:
    extension = _load()["stable_extension"]

    assert extension["namespace"] == "flagquantum.dynamic"
    assert list(fqd.__all__) == extension["additions"] == ["DynamicCircuit"]
    assert "DynamicCircuit" not in fq.__all__
    assert "DynamicCircuit" not in fq.experimental.dynamic.__all__


def test_dynamic_signatures_match_candidate() -> None:
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
    assert fqd.DynamicCircuit(1).x(0).state(refresh=True).shape == (1, 2)
    with pytest.raises(fqe.ValidationError):
        fqd.DynamicCircuit(1).measure(-1)
    with pytest.raises(fqe.ValidationError):
        fqd.DynamicCircuit(1).conditional("x", 0, conditions={0: 2})
    with pytest.raises(fqe.CapabilityError):
        fqd.DynamicCircuit(1).measure(0).state()


def test_native_dynamic_result_remains_experimental() -> None:
    assert "DynamicExecutionResult" in fq.experimental.dynamic.__all__
    assert "DynamicExecutionResult" not in fqd.__all__
    assert "run_dynamic" not in fqd.__all__
