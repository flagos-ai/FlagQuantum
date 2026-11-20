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
    ValueRef,
    verify_program_record,
)
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


@pytest.mark.parametrize("matrix", [None, ((1 + 0j, 1 + 0j), (0j, 1 + 0j))])
def test_verifier_rejects_missing_or_nonunitary_payload(
    matrix: tuple[tuple[complex, ...], ...] | None,
) -> None:
    q = ValueRef("entry", 0, "qubit")
    r = ValueRef("entry", 1, "qubit")
    op = Operation("any", (q,), (r,), matrix=matrix)
    record = ProgramRecord(
        (FunctionRecord("main", (BlockRecord("entry", (q,), operations=(op,)),)),)
    )
    assert "program.matrix_invalid" in {d.code for d in verify_program_record(record)}


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


def previous_inner_product_error(
    matrix: tuple[tuple[complex, ...], ...],
) -> str | None:
    """Keep the complete pre-optimization inner products as an independent oracle."""
    size = len(matrix)
    for left in range(size):
        for right in range(left, size):
            inner = sum(row[left].conjugate() * row[right] for row in matrix)
            if abs(inner - int(left == right)) > 1e-10:
                return "matrix must be unitary within absolute tolerance 1e-10"
    return None


def test_monomial_proof_matches_every_small_phased_permutation() -> None:
    from itertools import permutations, product

    for columns in permutations(range(4)):
        for phases in product((1 + 0j, -1 + 0j, 1j, -1j), repeat=4):
            matrix = tuple(
                tuple(phase if column == selected else 0j for column in range(4))
                for selected, phase in zip(columns, phases, strict=True)
            )
            assert matrix_error(matrix, 2) == previous_inner_product_error(matrix)


@pytest.mark.parametrize("offset", (0.0, 1e-12, 4e-11, 6e-11, 1e-8, -1e-8))
@pytest.mark.parametrize("duplicate", (False, True))
def test_monomial_invalid_columns_and_norm_tolerance_match_prior_validator(
    offset: float, duplicate: bool
) -> None:
    columns = (0, 1, 2, 2 if duplicate else 3)
    matrix = tuple(
        tuple(complex(1 + offset) if column == selected else 0j for column in range(4))
        for selected in columns
    )
    assert matrix_error(matrix, 2) == previous_inner_product_error(matrix)


def test_nonmonomial_fallback_matches_prior_validator() -> None:
    generator = torch.Generator().manual_seed(127)
    for width in (1, 2, 3):
        size = 1 << width
        dense = torch.randn((size, size), generator=generator, dtype=torch.complex128)
        unitary, _ = torch.linalg.qr(dense)
        for tensor in (dense, unitary, unitary * 1.0001, torch.zeros_like(unitary)):
            matrix = tuple(tuple(complex(value) for value in row) for row in tensor)
            assert matrix_error(matrix, width) == previous_inner_product_error(matrix)
    # The first rows can be monomial while a later block requires full products.
    root = 2**-0.5
    mixed = (
        (1 + 0j, 0j, 0j, 0j),
        (0j, 1j, 0j, 0j),
        (0j, 0j, complex(root), complex(root)),
        (0j, 0j, complex(root), complex(-root)),
    )
    assert matrix_error(mixed, 2) == previous_inner_product_error(mixed)
