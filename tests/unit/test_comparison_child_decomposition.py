"""Every concrete comparison source graph retains its recursive source glyphs."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_comparison_decomposition import EXACT, _record
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", EXACT)
def test_whole_comparison_file_recursive_definitions_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    record = _record(identifier)
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
            assert node.presentation is not None, (identifier, name, source)
            assert node.presentation[0] == _label(source)
            actual = node.score()
            assert isinstance(actual, dict)
            assert _canonical_score(actual) == _canonical_score(
                _source_score(composite)
            )
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )
