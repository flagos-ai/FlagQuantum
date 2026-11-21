"""Complete duplicate-detection source-family calls, typed views and fresh drawings."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.duplicate_decomposition import (
    duplicate_catalog_program,
    duplicate_score,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", ("has_duplicates", "has_duplicates_symb"))
def test_complete_factories_calls_typed_children_and_fresh_source_drawing(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    operation = native.name(identifier)
    record = duplicate_catalog_program(operation, ports)
    assert record is not None and verify_program_record(record) == ()
    composite = originals[1][identifier].decompose_bloq()
    # Named inputs are mappings. Source traversal groups their insertion order
    # by predecessor, while the native traversal uses the callee signature.
    for actual_call, source_call in zip(
        _native_trace(record), _source_trace(_source(ports), composite), strict=True
    ):
        assert isinstance(actual_call, tuple) and isinstance(source_call, tuple)
        assert (actual_call[0], dict(actual_call[1]), actual_call[2]) == (
            source_call[0],
            dict(source_call[1]),
            source_call[2],
        )
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
    actual = duplicate_score(operation, ports)
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)
    assert native.decomposition(identifier) == record
    assert native.score(identifier) == actual


def test_concrete_catalog_reuses_literal_owner_and_symbolic_calls_reject_binding() -> (
    None
):
    assert native.decomposition("has_duplicates") == sorted_duplicates_program(
        4, native.signature("has_duplicates")[0].dtype
    )
    record = native.decomposition("has_duplicates_symb")
    assert record is not None
    assert all(
        not f.blocks
        for f in record.functions
        if f.name in {"half_lt", "half_lt_inverse", "X"}
    )
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1,)))
