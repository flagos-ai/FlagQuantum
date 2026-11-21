"""Complete addition dependencies retain frozen source calls and fresh glyphs."""

from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_addition_decomposition import CONCRETE
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_comparison_decomposition import EXACT as COMPARISONS
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", (*CONCRETE, *COMPARISONS))
def test_complete_source_file_recursive_calls_and_child_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    record = native.decomposition(identifier)
    assert record is not None
    functions = {function.name: function for function in record.functions}
    entry = record.functions[0].name
    pending = [(entry, originals[1][identifier])]
    visited: dict[str, Any] = {}
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        try:
            composite = source.decompose_bloq()
        except (DecomposeTypeError, DecomposeNotImplementedError):
            continue
        assert _native_trace(record, name) == _source_trace(
            _source(Signature(function.registers or ())), composite
        )
        if name != entry:
            node = ComponentFunction(
                identifier, name, record, identifier, native._child_presentation
            )
            if node.presentation is None:
                assert identifier not in CONCRETE, (identifier, name, source)
            else:
                assert node.presentation[0] == _label(source)
                try:
                    expected = _source_score(composite)
                except IndexError:
                    assert identifier == "add_k_large" and _label(source) == "Add"
                else:
                    actual = node.score()
                    assert isinstance(actual, dict)
                    assert _canonical_score(actual) == _canonical_score(expected)
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )


def test_phase_mutation_rejects_addition_and_always_glyphs() -> None:
    record = native.decomposition("add_k_small")
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                function,
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(operation, opcode="tdg")
                                if operation.opcode == "t"
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        ),
    )
    qualified = [
        function.name
        for function in record.functions[1:]
        if tuple(r.name for r in function.registers or ()) == ("a", "b")
    ]
    assert len(qualified) == 2
    for entry in qualified:
        assert native._child_presentation(record, entry, "add_k_small") is not None
        node = ComponentFunction(
            "changed", entry, changed, "add_k_small", native._child_presentation
        )
        assert node.presentation is None
        assert "operation" not in node.attributes()


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("include_msb", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_out_of_place_child_both_directions_and_output_widths(
    width: int, include_msb: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    from examples.algorithms.addition_child_decomposition import (
        addition_child_decomposition,
    )
    from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
        out_of_place_addition_program,
    )
    from flagquantum.drawer.component_score import component_register_score
    from flagquantum.drawer.register_score_data import register_score_data

    record = out_of_place_addition_program(
        width, include_msb=include_msb, adjoint=adjoint
    )
    selected = addition_child_decomposition(record, record.functions[0].name, {})
    assert selected is not None
    label, template, symbols = selected
    assert label == "OutOfPlaceAdder"
    actual = register_score_data(
        component_register_score(template, template.functions[0].name),
        register_wire_symbol=symbols,
    )
    source = OutOfPlaceAdder(
        width, include_most_significant_bit=include_msb, is_adjoint=adjoint
    ).decompose_bloq()
    assert _canonical_score(dict(actual)) == _canonical_score(_source_score(source))
