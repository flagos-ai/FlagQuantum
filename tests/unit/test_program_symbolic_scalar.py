"""Owned symbolic leaves retain canonical payloads and reject malformed trees."""

from dataclasses import replace
from typing import Any

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    Operation,
    circuit_to_program,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.scalar import scalar_literal_error
from flagquantum.core.parameters import Parameter, ParameterExpression

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ("acos", "floor"))
def test_approved_unary_expression_preserves_program_and_ir_payload(
    operation: str,
) -> None:
    expression = ParameterExpression(operation, (Parameter("theta"),))
    circuit = CircuitIR(1, (Instruction("rz", (0,), {"theta": expression}),))
    lowered = circuit_to_program(circuit)
    assert lowered.state == "supported_exact" and lowered.record is not None
    assert verify_program_record(lowered.record) == ()
    constant = lowered.record.functions[0].blocks[0].operations[0]
    assert constant.literal == expression
    restored = program_to_circuit(lowered.record)
    assert restored.state == "supported_exact"
    assert restored.canonical_payload == circuit.to_dict()
    assert CircuitIR.from_json(circuit.to_json()).to_json() == circuit.to_json()


def test_symbolic_roundtrip_and_canonical_tamper_detection() -> None:
    theta = Parameter("theta")
    value = -(theta + 0.25) * (2 - Parameter("other"))
    circuit = CircuitIR(1, (Instruction("rz", (0,), {"theta": value}),))
    lowered = circuit_to_program(circuit)
    assert lowered.state == "supported_exact"
    assert lowered.record is not None
    record = lowered.record
    function = record.functions[0]
    block = function.blocks[0]
    constant = block.operations[0]
    assert constant.literal == value
    assert hash(constant) == hash(replace(constant))
    assert verify_program_record(record) == ()
    restored = program_to_circuit(record)
    assert restored.state == "supported_exact"
    assert restored.canonical_payload == circuit.to_dict()
    changed = replace(constant, literal=theta)
    altered = replace(
        record,
        functions=(
            replace(
                function,
                blocks=(replace(block, operations=(changed, *block.operations[1:])),),
            ),
        ),
    )
    assert verify_program_record(altered) == ()
    assert program_to_circuit(altered).state == "unsupported_with_diagnostics"


@pytest.mark.parametrize("value", [1, 0.25, torch.tensor(0.25, dtype=torch.float64)])
def test_concrete_normalization_and_hash_remain_compatible(value: Any) -> None:
    lowered = circuit_to_program(
        CircuitIR(1, (Instruction("rz", (0,), {"theta": value}),))
    )
    assert lowered.record is not None
    operation = lowered.record.functions[0].blocks[0].operations[0]
    assert type(operation.literal) is float
    assert operation.literal == float(value)
    assert hash(operation) == hash(
        ("constant", (), operation.results, (), float(value), ())
    )


@pytest.mark.parametrize(
    "value",
    [
        ParameterExpression("divide", (Parameter("x"), 2)),
        ParameterExpression("add", (Parameter("x"),)),
        ParameterExpression("neg", (Parameter("x"), 1)),
        ParameterExpression("add", (Parameter("x"), True)),
        ParameterExpression("add", (Parameter("x"), float("nan"))),
        ParameterExpression("add", (Parameter("x"), float("inf"))),
        ParameterExpression("mul", (Parameter("x"), object())),
        ParameterExpression("add", (Parameter("x"), 10**400)),
    ],
)
def test_malformed_symbolic_constants_have_program_diagnostics(
    value: ParameterExpression,
) -> None:
    valid = circuit_to_program(
        CircuitIR(1, (Instruction("rz", (0,), {"theta": Parameter("x")}),))
    )
    assert valid.record is not None
    function = valid.record.functions[0]
    block = function.blocks[0]
    changed = replace(block.operations[0], literal=value)
    record = replace(
        valid.record,
        functions=(
            replace(
                function,
                blocks=(replace(block, operations=(changed, *block.operations[1:])),),
            ),
        ),
    )
    assert "program.result_kind" in {d.code for d in verify_program_record(record)}
    assert program_to_circuit(record).state == "invalid_input"


def test_mutable_and_cyclic_trees_are_rejected_before_hashing() -> None:
    mutable: Any = [Parameter("x"), 1]
    assert scalar_literal_error(ParameterExpression("add", mutable)) is not None
    cyclic = ParameterExpression("neg", (1,))
    object.__setattr__(cyclic, "args", (cyclic,))
    assert scalar_literal_error(cyclic) == "scalar expression must be acyclic"
    malformed = Parameter("x")
    object.__setattr__(malformed, "name", "")
    assert scalar_literal_error(malformed) is not None
    assert scalar_literal_error(Operation("x")) is not None
