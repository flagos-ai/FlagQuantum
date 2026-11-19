"""Materialized channel Programs own and verify exact Kraus operators."""

from dataclasses import replace
from math import sqrt

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    circuit_to_program,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.kraus import kraus_error

pytestmark = pytest.mark.unit


def test_channel_roundtrip_copies_data_and_attests_every_operator() -> None:
    matrices = (
        torch.tensor([[1, 0], [0, sqrt(0.75)]], dtype=torch.complex128),
        torch.tensor([[0, 0.5j], [0, 0]], dtype=torch.complex128),
    )
    physical = CircuitIR(
        1, (Instruction("kraus", (0,), matrix=matrices, metadata={"is_channel": True}),)
    )
    result = circuit_to_program(physical)
    assert result.state == "supported_exact"
    assert result.record is not None
    record = result.record
    function = record.functions[0]
    block = function.blocks[0]
    operation = block.operations[0]
    assert operation.kraus is not None
    assert hash(operation) == hash(replace(operation))
    assert program_to_circuit(record).state == "supported_exact"
    matrices[0].zero_()
    assert operation.kraus[0][0][0] == 1 + 0j
    changed = replace(
        operation,
        kraus=tuple(
            tuple(tuple(-v for v in row) for row in matrix)
            for matrix in operation.kraus
        ),
    )
    altered = replace(
        record,
        functions=(replace(function, blocks=(replace(block, operations=(changed,)),)),),
    )
    assert verify_program_record(altered) == ()
    assert hash(changed) != hash(operation)
    assert program_to_circuit(altered).state == "unsupported_with_diagnostics"


@pytest.mark.parametrize(
    "operators",
    [
        None,
        (),
        [],
        (((1 + 0j,),),),
        (((1, 0), (0, 1)),),
        (((complex(float("nan")), 0j), (0j, 1 + 0j)),),
        (((1 + 0j, 0j), (0j, 0.5 + 0j)),),
        (((1e308 + 0j, 0j), (0j, 1e308 + 0j)),),
    ],
)
def test_invalid_channel_data_is_rejected(operators: object) -> None:
    assert kraus_error(operators, 1) is not None


def test_channel_payload_is_required_and_exclusive() -> None:
    q, out = ValueRef("entry", 0, "qubit"), ValueRef("entry", 1, "qubit")
    identity = (((1 + 0j, 0j), (0j, 1 + 0j)),)
    for operation in (
        Operation("kraus", (q,), (out,)),
        Operation("x", (q,), (out,), kraus=identity),
        Operation("kraus", (q,), (out,), kraus=identity, matrix=identity[0]),
    ):
        record = ProgramRecord(
            (
                FunctionRecord(
                    "main", (BlockRecord("entry", (q,), operations=(operation,)),)
                ),
            )
        )
        assert "program.matrix_invalid" in {
            d.code for d in verify_program_record(record)
        }
    assert kraus_error(identity, 0) is not None
    operation = Operation("x", (q,), (out,))
    assert hash(operation) == hash(("x", (q,), (out,), (), None, ()))


def test_channel_marker_is_required_at_the_physical_boundary() -> None:
    physical = CircuitIR(1, (Instruction("kraus", (0,), matrix=torch.eye(2)),))
    result = circuit_to_program(physical)
    assert result.state == "unsupported_with_diagnostics"
    assert result.diagnostics[0].code == "program.matrix_invalid"


@pytest.mark.parametrize(
    "matrices", [(), (torch.ones(2),), (torch.ones(2, 3),), (torch.eye(2) / 2,)]
)
def test_malformed_physical_channel_reports_diagnostic(matrices: object) -> None:
    physical = CircuitIR(
        1, (Instruction("kraus", (0,), matrix=matrices, metadata={"is_channel": True}),)
    )
    result = circuit_to_program(physical)
    assert result.state == "unsupported_with_diagnostics"
    assert result.diagnostics[0].code == "program.matrix_invalid"
