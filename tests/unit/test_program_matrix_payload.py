"""Owned matrix semantics, validation and tamper detection at Program level."""

from dataclasses import replace

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
)
from flagquantum.core.ir.program import ValueRef, verify_program_record
from flagquantum.core.ir.program.lowering import circuit_to_program, program_to_circuit
from flagquantum.core.ir.program.matrix import matrix_error

pytestmark = pytest.mark.unit


def test_matrix_lowering_copies_entries_and_attests_full_phase() -> None:
    matrix = torch.tensor([[0, 1j], [1j, 0]], dtype=torch.complex128)
    circuit = CircuitIR(
        n_wires=1, instructions=(Instruction("any", (0,), matrix=matrix),)
    )
    conversion = circuit_to_program(circuit)
    assert conversion.state == "supported_exact"
    record = conversion.record
    assert record is not None
    block = record.functions[0].blocks[0]
    operation = block.operations[0]
    assert operation.matrix == ((0j, 1j), (1j, 0j))
    assert not verify_program_record(record)
    assert program_to_circuit(record).state == "supported_exact"
    assert hash(operation) == hash(replace(operation))
    altered = replace(operation, matrix=((0j, -1j), (-1j, 0j)))
    tampered = replace(
        record,
        functions=(
            replace(
                record.functions[0], blocks=(replace(block, operations=(altered,)),)
            ),
        ),
    )
    assert not verify_program_record(tampered)
    assert program_to_circuit(tampered).state == "unsupported_with_diagnostics"
    matrix.zero_()
    assert operation.matrix == ((0j, 1j), (1j, 0j))


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        ((1j,),),
        ((1 + 0j, 0j),),
        [[1 + 0j, 0j], [0j, 1 + 0j]],
        ((1, 0), (0, 1)),
        ((complex("nan"), 0j), (0j, 1 + 0j)),
        ((complex("inf"), 0j), (0j, 1 + 0j)),
        ((1 + 0j, 1 + 0j), (0j, 1 + 0j)),
    ],
)
def test_malformed_matrix_has_typed_diagnostic(matrix: object) -> None:
    assert matrix_error(matrix, 1) is not None


def test_payload_on_other_opcode_and_zero_arity_are_rejected() -> None:
    q = ValueRef("entry", 0, "qubit")
    r = ValueRef("entry", 1, "qubit")
    matrix = ((0j, 1j), (1j, 0j))
    op = Operation("x", (q,), (r,), matrix=matrix)
    record = ProgramRecord(
        (FunctionRecord("main", (BlockRecord("entry", (q,), operations=(op,)),)),)
    )
    assert "program.matrix_invalid" in {d.code for d in verify_program_record(record)}
    assert matrix_error(((1 + 0j,),), 0) is not None
    plain = replace(op, matrix=None)
    assert hash(plain) == hash(
        (
            plain.opcode,
            plain.operands,
            plain.results,
            plain.parameters,
            plain.literal,
            plain.condition_clauses,
        )
    )
