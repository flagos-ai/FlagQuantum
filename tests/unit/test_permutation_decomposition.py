"""Complete permutation source-family calls, typed views and fresh drawings."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.permutation_decomposition import (
    permutation_catalog_program,
    permutation_score,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier",
    (
        "permutation",
        "permutation_symb_with_cycles",
        "sparse_permutation",
        "sparse_permutation_with_symbolic_N",
    ),
)
def test_complete_factories_calls_typed_children_and_fresh_source_drawing(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    operation = native.name(identifier)
    record = permutation_catalog_program(
        operation, ports, native.attributes(identifier)
    )
    assert record is not None and verify_program_record(record) == ()
    composite = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), composite)
    functions = {f.name: f for f in record.functions}
    calls = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
    encodings = {"QAny": "opaque", "QBit": "bit", "QUInt": "uint", "BQUInt": "uint"}
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
        for actual_port, source_port in zip(
            child.registers or (), calls[i].signature, strict=True
        ):
            assert (
                dict(actual_port.dtype.parameters)["bound"]
                == source_port.dtype.iteration_length
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
    actual = permutation_score(operation, ports, native.attributes(identifier))
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)
    assert native.decomposition(identifier) == record
    assert native.score(identifier) == actual


@pytest.mark.parametrize("identifier", ("permutation", "sparse_permutation"))
def test_concrete_catalog_reuses_exact_owner(identifier: str) -> None:
    parameters = native.attributes(identifier)
    size, cycles, width = (
        parameters["N"],
        parameters["cycles"],
        native.signature(identifier)[0].bitsize,
    )
    assert type(size) is int and type(width) is int and isinstance(cycles, tuple)
    assert native.decomposition(identifier) == permutation_program(size, cycles, width)


@pytest.mark.parametrize(
    "identifier", ("permutation_symb_with_cycles", "sparse_permutation_with_symbolic_N")
)
def test_symbolic_cycle_definitions_remain_nonexecuting(identifier: str) -> None:
    record = native.decomposition(identifier)
    assert record is not None
    leaves = [f for f in record.functions[1:] if not f.blocks]
    assert leaves
    if identifier == "permutation_symb_with_cycles":
        assert all(not f.blocks for f in record.functions[1:])
    else:
        assert any(f.blocks for f in record.functions[1:])
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,),))


@pytest.mark.parametrize(
    "identifier",
    (
        "permutation_symb",
        "permutation_cycle_symb",
        "permutation_cycle_symb_N",
        "permutation_cycle",
    ),
)
def test_original_refusals_remain_explicit_atomic_boundaries(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.drawing import get_musical_score_data

    source = originals[1][identifier]
    if identifier in {"permutation_symb", "permutation_cycle_symb"}:
        with pytest.raises(DecomposeTypeError):
            source.decompose_bloq()
    else:
        with pytest.raises(ValueError, match="should return a `Text` object"):
            get_musical_score_data(source.decompose_bloq())
    if identifier in {"permutation_symb", "permutation_cycle_symb"}:
        assert native.decomposition(identifier) is None
    else:
        assert native.decomposition(identifier) is not None
        assert (
            permutation_score(
                native.name(identifier),
                native.signature(identifier),
                native.attributes(identifier),
            )
            is None
        )
