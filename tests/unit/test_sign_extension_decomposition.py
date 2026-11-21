"""Complete sign-extension source-family calls, typed views and fresh drawings."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.sign_extension_decomposition import (
    sign_extension_catalog_program,
    sign_extension_score,
)
from flagquantum.compiler.registers.bits.sign_extension_component import (
    sign_extension_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", ("sign_extend", "sign_truncate"))
def test_complete_factories_calls_typed_children_and_fresh_source_drawing(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    operation = native.name(identifier)
    record = sign_extension_catalog_program(operation, ports)
    assert record is not None and verify_program_record(record) == ()
    truncate = identifier == "sign_truncate"
    narrow, wide = (ports[1], ports[0]) if truncate else tuple(ports)
    assert record == sign_extension_program(narrow.dtype, wide.dtype, adjoint=truncate)
    composite = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), composite)
    functions = {f.name: f for f in record.functions}
    calls = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
    encodings = {"QAny": "opaque", "QBit": "bit", "QInt": "signed"}
    operations = record.functions[0].blocks[0].operations
    assert len(operations) == len(calls) == 5
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
    if truncate:
        assert ports[1].dtype.encoding == "signed"
        assert calls[4].signature.get_right("reg").dtype.__class__.__name__ == "QAny"
        assert (functions[operations[4].callee or ""].registers or ())[
            -1
        ].dtype.encoding == "opaque"
    fresh = get_musical_score_data(composite)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    actual = sign_extension_score(operation, ports)
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)
    assert native.decomposition(identifier) == record
    assert native.score(identifier) == actual
