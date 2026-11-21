"""Complete sorting source graphs retain every recursive source drawing."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier", ("comparator", "parallel_compare", "bitonic_merge", "bitonic_sort")
)
def test_whole_sorting_file_recursive_definitions_and_fresh_drawings(
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


@pytest.mark.parametrize(
    "count,offset", ((3, 1), (3, 2), (4, 1), (4, 2), (5, 3), (7, 2))
)
def test_parallel_offset_comes_from_operands(count: int, offset: int) -> None:
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    from examples.algorithms.sorting_child_decomposition import (
        sorting_child_decomposition,
    )
    from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
        parallel_comparison_program,
    )
    from flagquantum.drawer.component_score import component_register_score
    from flagquantum.drawer.register_score_data import register_score_data

    record = parallel_comparison_program(count, offset, 2)
    presentation = sorting_child_decomposition(record, record.functions[0].name)
    assert presentation is not None
    label, template, symbols = presentation
    assert label == "ParallelComparators"
    assert template == record
    actual = register_score_data(
        component_register_score(template, template.functions[0].name),
        register_wire_symbol=symbols,
    )
    assert _canonical_score(dict(actual)) == _canonical_score(
        _source_score(ParallelComparators(count, offset, 2).decompose_bloq())
    )


@pytest.mark.parametrize(
    "identifier", ("comparator", "parallel_compare", "bitonic_merge", "bitonic_sort")
)
def test_feedback_mutation_rejects_the_complete_sorting_wrapper(
    identifier: str,
) -> None:
    from dataclasses import replace

    from examples.algorithms.sorting_child_decomposition import (
        sorting_child_decomposition,
    )

    record = native.decomposition(identifier)
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
                            replace(
                                op,
                                condition_clauses=tuple(
                                    tuple(
                                        replace(c, equals=not c.equals) for c in clause
                                    )
                                    for clause in op.condition_clauses
                                ),
                            )
                            for op in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        ),
    )
    assert changed != record
    assert sorting_child_decomposition(changed, changed.functions[0].name) is None


def test_ambiguous_two_word_body_requires_a_matching_recursive_parent() -> None:
    from examples.algorithms.sorting_child_decomposition import (
        sorting_child_decomposition,
    )
    from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
        parallel_comparison_program,
    )
    from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
        bitonic_sort_program,
    )

    for record in (parallel_comparison_program(2, 1, 2), bitonic_sort_program(2, 2)):
        assert sorting_child_decomposition(record, record.functions[0].name) is None
