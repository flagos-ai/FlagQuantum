"""Complete subtraction source-family catalog calls and fresh original drawings."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.subtraction_decomposition import (
    subtraction_catalog_program,
    subtraction_score,
)
from flagquantum.compiler.registers.arithmetic.subtraction_component import (
    subtract_from_program,
    subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONCRETE = (
    "sub_small",
    "sub_large",
    "sub_diff_size_regs",
    "sub_from_small",
    "sub_from_large",
)
SYMBOLIC = ("sub_symb", "sub_from_symb")


@pytest.mark.parametrize("identifier", (*CONCRETE, *SYMBOLIC))
def test_every_successful_original_factory_ports_calls_operands_and_glyphs(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    record = subtraction_catalog_program(native.name(identifier), ports)
    assert record is not None and verify_program_record(record) == ()
    composite = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), composite)
    functions = {f.name: f for f in record.functions}
    source_calls = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
    encodings = {"QAny": "opaque", "QBit": "bit", "QInt": "signed", "QUInt": "uint"}
    for i, operation in enumerate(record.functions[0].blocks[0].operations):
        function = functions[operation.callee or ""]
        actual_ports = tuple(
            (r.name, r.bitsize, r.shape_symbolic, r.side.value, r.dtype.encoding)
            for r in function.registers or ()
        )
        expected_ports = tuple(
            (
                r.name,
                r.bitsize,
                r.shape,
                r.side.value,
                encodings[type(r.dtype).__name__],
            )
            for r in source_calls[i].signature
        )
        assert actual_ports == expected_ports
    fresh = get_musical_score_data(composite)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    actual = subtraction_score(native.name(identifier), ports)
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)


@pytest.mark.parametrize("identifier", CONCRETE)
def test_concrete_program_is_exact_existing_executable_owner(identifier: str) -> None:
    ports = native.signature(identifier)
    expected = (
        subtract_from_program(ports[0].dtype)
        if identifier.startswith("sub_from")
        else subtraction_program(ports[0].dtype, ports[1].dtype)
    )
    assert subtraction_catalog_program(native.name(identifier), ports) == expected


@pytest.mark.parametrize("identifier", SYMBOLIC)
def test_symbolic_graph_retains_source_refusal_leaves_and_physical_rejection(
    identifier: str,
) -> None:
    record = subtraction_catalog_program(
        native.name(identifier), native.signature(identifier)
    )
    assert record is not None
    declared = [f for f in record.functions[1:] if not f.blocks]
    expected = {("a", "b"), ("q",)}
    if identifier == "sub_symb":
        expected.add(("reg", "reg"))
    assert {tuple(r.name for r in f.registers or ()) for f in declared} == expected
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1,)))


def test_original_composite_decompose_failure_remains_explicit(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="CompositeBloq is ill-defined"):
        originals[1]["sub_symp_decomposition"].decompose_bloq()
    assert (
        subtraction_catalog_program(
            native.name("sub_symp_decomposition"),
            native.signature("sub_symp_decomposition"),
        )
        is None
    )
