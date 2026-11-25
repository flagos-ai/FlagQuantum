"""Import once and call repeatedly without changing ownership or physical IR."""

from itertools import product

import pytest

from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _two_additions(explicit_import: bool) -> ProgramRecord:
    word = unsigned_register_type(3)
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=word))
    values = builder.inputs
    for constant in (1, 2, 1):
        record = constant_addition_program(word, constant)
        if explicit_import:
            callee = builder.import_program(record)
            assert builder.import_program(record) == callee
            values = builder.call_dict(callee, values)
        else:
            values = builder.call_program(record, values)
    return builder.finalize(values)


def test_explicit_import_preserves_conflicting_names_full_records_and_every_branch() -> (
    None
):
    ordinary = _two_additions(False)
    imported = _two_additions(True)
    assert imported == ordinary
    bound = bind_register_program(imported, ((0, 1, 2),), clean=tuple(range(3, 12)))
    assert bound == bind_register_program(
        ordinary, ((0, 1, 2),), clean=tuple(range(3, 12))
    )
    width = bound.circuit.n_wires
    before = tuple(v << (width - 3) for v in range(8))
    after = tuple(((v + 4) % 8) << (width - 3) for v in range(8))
    inputs = tuple(
        tuple((v >> (width - 1 - i)) & 1 for i in range(width)) for v in before
    )
    expected = tuple(
        tuple((v >> (width - 1 - i)) & 1 for i in range(width)) for v in after
    )
    mapped, measured = certified_and_channel_map(bound.circuit.instructions, inputs)
    assert mapped == expected
    assert measured <= 8
    weights = [complex(i + 1, 3 - 2 * i) for i in range(8)]
    norm = sum(abs(v) ** 2 for v in weights) ** 0.5
    initial = {k: v / norm for k, v in zip(before, weights, strict=True)}
    desired = {
        k: v / norm * 2 ** (-measured / 2) for k, v in zip(after, weights, strict=True)
    }
    for history in product((0, 1), repeat=measured):
        actual = _branch(bound.circuit.instructions, width, history, initial.copy())
        assert {k: v for k, v in actual.items() if abs(v) > 1e-20} == pytest.approx(
            desired, abs=2e-14, rel=0
        )


def test_import_only_emits_no_call_and_does_not_consume_input() -> None:
    word = unsigned_register_type(3)
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=word))
    child = constant_addition_program(word, 1)
    callee = builder.import_program(child)
    assert builder.import_program(child) == callee
    with pytest.raises(ValueError, match="unknown or untyped"):
        builder.import_program(child, entry="absent")
    record = builder.finalize(builder.inputs)
    assert record.functions[0].blocks[0].operations == ()
    bound = bind_register_program(record, ((2, 0, 1),))
    assert bound.circuit.instructions == ()
    assert bound.outputs == ((2, 0, 1),) and bound.events == ()
