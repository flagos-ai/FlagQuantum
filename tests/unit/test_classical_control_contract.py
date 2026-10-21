from __future__ import annotations

from dataclasses import fields

import pytest

import flagquantum as fq
import flagquantum.core as fqc
import flagquantum.errors as fqe
import flagquantum.runtime.execution as runtime_execution
from flagquantum.core.classical_control import (
    CLASSICAL_CONTROL_KEYS,
    is_classical_control_instruction,
    validate_classical_control,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit


def _condition(bit: int) -> Instruction:
    return Instruction("x", (0,), metadata={"conditions": ((bit, 1),)})


def test_classical_control_contract_is_internal_and_schema_neutral() -> None:
    assert CLASSICAL_CONTROL_KEYS == (
        "is_dynamic",
        "classical_bit",
        "conditions",
        "condition_clauses",
    )
    assert "CLASSICAL_CONTROL_KEYS" not in fq.__all__
    assert "CLASSICAL_CONTROL_KEYS" not in getattr(fqc, "__all__", ())
    assert [item.name for item in fields(CircuitIR)] == [
        "n_wires",
        "instructions",
        "version",
        "dtype",
        "shape",
        "observables",
        "measurements",
        "metadata",
    ]
    assert [item.name for item in fields(Instruction)] == [
        "name",
        "wires",
        "params",
        "matrix",
        "metadata",
    ]
    assert [item.name for item in fields(MeasurementNode)] == [
        "kind",
        "wires",
        "shots",
        "metadata",
    ]
    assert [item.name for item in fields(ObservableNode)] == [
        "name",
        "wires",
        "coefficient",
        "metadata",
    ]


@pytest.mark.parametrize(
    "metadata",
    (
        {"is_dynamic": False},
        {"classical_bit": 0},
        {"conditions": ()},
        {"condition_clauses": ()},
    ),
)
def test_classical_control_detection_is_by_key_presence(metadata) -> None:
    assert is_classical_control_instruction(Instruction("x", (0,), metadata=metadata))


def test_classical_control_validator_returns_register_width() -> None:
    instructions = (
        Instruction(
            "measure",
            (0,),
            metadata={"is_dynamic": True, "classical_bit": 3},
        ),
        _condition(3),
    )
    assert validate_classical_control(instructions, n_qubits=1) == 4


@pytest.mark.parametrize(
    ("build", "message"),
    (
        (
            lambda: DynamicCircuit(1)
            .measure(0, classical_bit=0)
            .conditional("x", 0, classical_bit=1),
            "classical bit 1 was read before measurement",
        ),
        (
            lambda: DynamicCircuit(1).conditional("x", 0, classical_bit=0),
            "classical bit 0 was read before measurement",
        ),
        (
            lambda: DynamicCircuit(1)
            .measure(0, classical_bit=1)
            .conditional("x", 0, classical_bit=0),
            "classical bit 0 was read before measurement",
        ),
    ),
)
def test_classical_control_def_use_is_validated_at_build_time(build, message) -> None:
    with pytest.raises(fqe.ValidationError, match=message):
        build()


@pytest.mark.parametrize(
    ("metadata", "error", "message"),
    (
        (
            {"conditions": ((0, 1),), "condition_clauses": (((0, 1),),)},
            fqe.ValidationError,
            "both conditions and condition_clauses",
        ),
        ({"classical_bit": -1}, fqe.ValidationError, "non-negative"),
        (
            {"classical_bit": 0},
            fqe.ValidationError,
            "only on a measure instruction",
        ),
        ({"conditions": ((0, 2),)}, fqe.ValidationError, "zero or one"),
        ({"conditions": ((0, True),)}, TypeError, "must be an integer"),
        ({"conditions": ((1, 1), (0, 1))}, fqe.ValidationError, "ordered"),
        (
            {"condition_clauses": (((1, 1),), ((0, 1),))},
            fqe.ValidationError,
            "canonical, distinct, and minimal",
        ),
    ),
)
def test_classical_control_metadata_fails_closed(metadata, error, message) -> None:
    with pytest.raises(error, match=message):
        validate_classical_control(
            (Instruction("x", (0,), metadata=metadata),),
            n_qubits=1,
        )


def test_classical_control_measure_requires_a_result_bit() -> None:
    with pytest.raises(fqe.ValidationError, match="requires classical_bit"):
        validate_classical_control(
            (Instruction("measure", (0,), metadata={"is_dynamic": True}),),
            n_qubits=1,
        )


@pytest.mark.parametrize(
    "metadata",
    (
        {"is_dynamic": False},
        {"classical_bit": 0},
        {"conditions": ((0, 1),)},
        {"condition_clauses": (((0, 1),),)},
    ),
)
def test_distributed_execution_refuses_every_classical_control_key(
    monkeypatch, metadata
) -> None:
    def compile_must_not_run(*args, **kwargs):
        raise AssertionError("dynamic IR reached compilation")

    monkeypatch.setattr(runtime_execution, "compile_program", compile_must_not_run)
    ir = CircuitIR(1, (Instruction("x", (0,), metadata=metadata),))

    with pytest.raises(
        fqe.CapabilityError,
        match="dynamic_conditional_execution_unsupported.*run_dynamic",
    ):
        runtime_execution.run_distributed(ir)
