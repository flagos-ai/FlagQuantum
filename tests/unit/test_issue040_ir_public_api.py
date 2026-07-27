import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import (
    IR_VERSION,
    CircuitIR,
    Instruction,
    IRSerializationError,
    IRValidationError,
    MeasurementNode,
    ObservableNode,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Instruction("", (0,)), "opcode cannot be empty"),
        (lambda: Instruction("cx", (0,)), "requires 2 wire"),
        (lambda: Instruction("rx", (0,)), "missing parameter"),
        (lambda: Instruction("unknown", (0,)), "custom operations require"),
        (lambda: CircuitIR(0, ()), "n_wires must be positive"),
        (
            lambda: CircuitIR(2, (Instruction("cx", (0, 2)),)),
            "outside circuit range",
        ),
        (
            lambda: MeasurementNode("sample", (0,), shots=0),
            "shots must be a positive",
        ),
    ],
)
def test_invalid_ir_fails_with_actionable_errors(factory, message):
    with pytest.raises((IRValidationError, ValueError), match=message):
        factory()


def test_circuit_rejects_out_of_bounds_gate_at_construction():
    with pytest.raises(ValueError, match="outside circuit range"):
        fq.Circuit(2).cx(0, 2)


def test_ir_v1_round_trip_is_deterministic_and_hashes_all_content():
    theta = fq.Parameter("theta")
    ir = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("rx", (1,), params={"theta": theta + 0.25}),
            Instruction("custom", (0,), matrix=torch.eye(2, dtype=torch.complex64)),
        ),
        dtype="complex64",
        shape=(1, 4),
        observables=(ObservableNode("pauli_z", (1,), coefficient=0.5),),
        measurements=(MeasurementNode("sample", (0, 1), shots=100),),
        metadata={"seed": 7},
    )

    text = ir.to_json()
    restored = CircuitIR.from_json(text)

    assert restored.version == IR_VERSION == "1.0"
    assert restored.to_json() == text
    assert restored.content_hash == ir.content_hash
    assert len(restored.content_hash) == 64
    assert restored.dtype == "complex64"
    assert restored.shape == (1, 4)
    assert restored.observables[0].name == "pauli_z"
    assert restored.measurements[0].shots == 100
    assert torch.equal(
        restored.instructions[2].matrix, torch.eye(2, dtype=torch.complex64)
    )

    changed = CircuitIR.from_dict({**ir.to_dict(), "metadata": {"seed": 8}})
    assert changed.content_hash != ir.content_hash


def test_serialized_ir_rejects_wrong_kind_version_and_invalid_json():
    payload = fq.Circuit(1).h(0).to_ir().to_dict()
    with pytest.raises(IRSerializationError, match="not a FlagQuantum"):
        CircuitIR.from_dict({**payload, "kind": "other"})
    with pytest.raises(IRValidationError, match="unsupported IR version"):
        CircuitIR.from_dict({**payload, "version": "99.0"})
    with pytest.raises(IRSerializationError, match="invalid CircuitIR JSON"):
        CircuitIR.from_json("{")


def test_executor_boundary_rejects_non_ir_input_before_dispatch():
    class InvalidProgram:
        def to_ir(self):
            return {"n_wires": 2}

    with pytest.raises(TypeError, match="executors require CircuitIR"):
        fq.run_native(InvalidProgram())


def test_public_api_snapshot_and_experimental_deprecation_path():
    snapshot = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    assert sorted(fq.__all__) == snapshot["stable_exports"]
    assert "JAXStatevectorShardState" not in fq.__all__
    assert fq.experimental.JAXStatevectorShardState is not None
    with pytest.warns(DeprecationWarning, match="flagquantum.experimental"):
        assert fq.JAXStatevectorShardState is fq.experimental.JAXStatevectorShardState


def test_large_mps_circuit_ir_uses_symbolic_amplitude_shape():
    ir = fq.Circuit(16_384).ry(0, theta=0.1).to_ir()

    assert ir.shape == (1,)
    assert ir.metadata["logical_state_shape"] == {
        "batch_size": 1,
        "amplitude_dimension": "power_of_two",
        "amplitude_exponent": 16_384,
    }
    assert len(ir.content_hash) == 64
