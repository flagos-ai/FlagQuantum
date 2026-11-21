"""Complete duplicate and conditional arithmetic source files retain typed calls."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

EXACT = (
    "has_duplicates",
    "has_duplicates_symb",
    "ctrl_add_or_sub_signed_symb",
    "ctrl_add_or_sub_unsigned",
    "ctrl_add_or_sub_signed",
)
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", EXACT)
def test_complete_recursive_named_calls_class_labels_and_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    record = native.decomposition(identifier)
    assert record is not None
    functions = {f.name: f for f in record.functions}
    pending = [(record.functions[0].name, originals[1][identifier])]
    visited: dict[str, Any] = {}
    for name, source in pending:
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        try:
            composite = source.decompose_bloq()
        except (DecomposeNotImplementedError, DecomposeTypeError):
            continue
        actual = _native_trace(record, name)
        expected = _source_trace(
            _source(Signature(function.registers or ())), composite
        )
        # Named inputs are mappings. Source iteration groups keys by predecessor.
        for left, right in zip(actual, expected, strict=True):
            assert isinstance(left, tuple) and isinstance(right, tuple)
            assert (left[0], dict(left[1]), left[2]) == (
                right[0],
                dict(right[1]),
                right[2],
            )
        if name != record.functions[0].name:
            node = ComponentFunction(
                identifier, name, record, identifier, native._child_presentation
            )
            assert node.presentation is not None, (identifier, source, name)
            assert node.presentation[0] == _label(source)
            drawing = node.score()
        else:
            drawing = native.score(identifier)
        assert isinstance(drawing, dict)
        try:
            original_score = _source_score(composite)
        except ValueError as error:
            assert name == record.functions[0].name
            assert type(source).__name__ == "ControlledAddOrSubtract"
            assert "should return a `Text` object" in str(error)
        else:
            assert _canonical_score(drawing) == _canonical_score(original_score), (
                identifier,
                source,
            )
        children = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
        pending.extend(
            (op.callee or "", children[i])
            for i, op in enumerate(function.blocks[0].operations)
        )


@pytest.mark.parametrize("identifier", EXACT[2:])
def test_conditional_adjoint_factory_retains_original_symbolic_and_concrete_calls(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from examples.algorithms.conditional_addition_decomposition import (
        conditional_addition_catalog_program,
    )

    ports = native.signature(identifier)
    record = conditional_addition_catalog_program(
        native.name(identifier),
        ports,
        {**native.attributes(identifier), "add_when_ctrl_is_on": False},
    )
    assert record is not None
    assert _native_trace(record) == _source_trace(
        _source(ports), originals[1][identifier].adjoint().decompose_bloq()
    )


def test_controlled_complement_labels_reject_changed_cnot_body() -> None:
    from dataclasses import replace

    from examples.algorithms.controlled_word_decomposition import (
        controlled_word_child_decomposition,
    )

    record = native.decomposition("ctrl_add_or_sub_unsigned")
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            replace(op, opcode="cz") if op.opcode == "cx" else op
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    assert changed != record
    rejected = set()
    for function in record.functions:
        if controlled_word_child_decomposition(record, function.name) is not None:
            assert controlled_word_child_decomposition(changed, function.name) is None
            rejected.add(tuple(r.name for r in function.registers or ()))
    assert rejected == {("ctrl", "x"), ("ctrl", "q")}
