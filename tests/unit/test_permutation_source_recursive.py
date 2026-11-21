"""Complete permutation source files retain finite recursive calls at symbolic widths."""

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
    "permutation",
    "permutation_symb_with_cycles",
    "sparse_permutation",
    "sparse_permutation_with_symbolic_N",
    "permutation_cycle",
    "permutation_cycle_symb_N",
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", EXACT)
def test_complete_permutation_recursive_graph(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    record = native.decomposition(identifier)
    assert record is not None
    functions = {function.name: function for function in record.functions}
    pending = [(record.functions[0].name, originals[1][identifier])]
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
        if name != record.functions[0].name:
            node = ComponentFunction(
                identifier, name, record, identifier, native._child_presentation
            )
            assert node.presentation is not None, (identifier, source, name)
            assert node.presentation[0] == _label(source)
            actual = node.score()
            assert isinstance(actual, dict)
            try:
                expected = _source_score(composite)
            except ValueError as error:
                assert type(source).__name__ == "PermutationCycle"
                assert "should return a `Text` object" in str(error)
            else:
                assert _canonical_score(actual) == _canonical_score(expected), (
                    identifier,
                    source,
                )
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )
