"""Whole bitwise and negation catalog decompositions retain source ownership."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.word_arithmetic_decomposition import word_program, word_score
from flagquantum.compiler.registers.arithmetic.negation_component import (
    negation_program,
)
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier", ("negate", "negate_symb", "xor", "bitwise_not", "bitwise_not_symb")
)
def test_complete_source_immediate_calls_and_fresh_drawings(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    record = word_program(native.name(identifier), ports)
    assert record is not None and verify_program_record(record) == ()
    source = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), source)
    fresh = get_musical_score_data(source)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    actual = word_score(native.name(identifier), ports)
    assert isinstance(actual, dict)
    assert _canonical_score(dict(actual)) == _canonical_score(expected)


@pytest.mark.parametrize("identifier", ("negate", "xor", "bitwise_not"))
def test_concrete_graph_is_exact_existing_executable_owner(identifier: str) -> None:
    ports = native.signature(identifier)
    record = word_program(native.name(identifier), ports)
    dtype = ports[0].dtype
    expected = (
        negation_program(dtype)
        if identifier == "negate"
        else bitwise_word_program(
            dtype, "xor" if identifier == "xor" else "bitwise_not"
        )
    )
    assert record == expected


@pytest.mark.parametrize("identifier", ("negate_symb", "bitwise_not_symb"))
def test_symbolic_graphs_retain_source_refusal_leaves_and_physical_rejection(
    identifier: str,
) -> None:
    record = word_program(native.name(identifier), native.signature(identifier))
    assert record is not None
    declared = [f for f in record.functions[1:] if not f.blocks]
    expected = {("q",), ("x",)} if identifier == "negate_symb" else {("q",)}
    assert {tuple(r.name for r in f.registers or ()) for f in declared} == expected
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,),))


def test_symbolic_xor_retains_unchanged_source_failure(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError, match="concrete value"):
        originals[1]["xor_symb"].decompose_bloq()
    assert word_program("Xor", native.signature("xor_symb")) is None
    assert word_program("Add", native.signature("add_small")) is None
