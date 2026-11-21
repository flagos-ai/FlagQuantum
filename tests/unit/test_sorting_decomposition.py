"""Complete sorting source-family calls, typed views and fresh drawings."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.sorting_decomposition import (
    sorting_catalog_program,
    sorting_score,
)
from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.bitonic_merge_component import (
    bitonic_merge_program,
)
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier",
    (
        "comparator",
        "comparator_symb",
        "parallel_compare",
        "bitonic_merge",
        "bitonic_sort",
        "bitonic_sort_symb",
    ),
)
def test_complete_factories_calls_typed_children_and_fresh_source_drawing(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    operation = native.name(identifier)
    record = sorting_catalog_program(operation, ports, native.attributes(identifier))
    assert record is not None and verify_program_record(record) == ()
    composite = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), composite)
    functions = {f.name: f for f in record.functions}
    calls = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
    encodings = {"QAny": "opaque", "QBit": "bit", "QUInt": "uint"}
    operations = record.functions[0].blocks[0].operations
    assert len(operations) == len(calls)
    for i, call in enumerate(operations):
        child = functions[call.callee or ""]
        assert tuple(
            (r.name, r.bitsize, r.shape_symbolic, r.side.value, r.dtype.encoding)
            for r in child.registers or ()
        ) == tuple(
            (
                r.name,
                r.bitsize,
                r.shape,
                r.side.value,
                encodings[type(r.dtype).__name__],
            )
            for r in calls[i].signature
        )
    fresh = get_musical_score_data(composite)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    actual = sorting_score(operation, ports, native.attributes(identifier))
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)
    assert native.decomposition(identifier) == record
    assert native.score(identifier) == actual


@pytest.mark.parametrize(
    "identifier", ("comparator", "parallel_compare", "bitonic_merge", "bitonic_sort")
)
def test_concrete_catalog_exactly_reuses_existing_executable_owner(
    identifier: str,
) -> None:
    expected = {
        "comparator": lambda: compare_exchange_program(3),
        "parallel_compare": lambda: parallel_comparison_program(7, 2, 3),
        "bitonic_merge": lambda: bitonic_merge_program(4, 7),
        "bitonic_sort": lambda: bitonic_sort_program(8, 4),
    }
    assert native.decomposition(identifier) == expected[identifier]()


@pytest.mark.parametrize("identifier", ("comparator_symb", "bitonic_sort_symb"))
def test_symbolic_calls_preserve_declared_children_and_reject_physical_binding(
    identifier: str,
) -> None:
    record = native.decomposition(identifier)
    assert record is not None
    declared = [function for function in record.functions[1:] if not function.blocks]
    assert {tuple(r.name for r in f.registers or ()) for f in declared} == {
        ("ctrl", "x", "y"),
        ("x", "y", "target"),
    }
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1,)))
