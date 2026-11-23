"""Sorting wire constructors retain owned networks and logical histories."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_sorting_factories import sorting_factories
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_sorting_component import sorting_case
from tests.unit.test_sorting_declaration import native_sorting_declaration

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PREFIX = "qualtran.bloqs.arithmetic.sorting."


@pytest.mark.parametrize("kind", ("compare", "parallel", "merge", "sort"))
@pytest.mark.parametrize("width", (1, 2, 3))
def test_original_wire_reuses_exact_program_bound_operands_and_resource_inventory(
    kind: str, width: int
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    expected, source, count, histories = sorting_case(kind, width)
    program = resolve_component_program(_library(source, 0), 0, sorting_factories())
    assert verify_program_record(program) == ()
    assert program == _rename_program(expected, program.functions[0].name)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    clean = tuple(range(count * width, count * width + histories + 3 * width))
    bound = bind_register_program(
        program, words, clean=clean, allow_workspace_outputs=True
    )
    assert bound == bind_register_program(
        expected, words, clean=clean, allow_workspace_outputs=True
    )
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == model.measurement
    assert len(bound.outputs) == count + histories


@pytest.mark.parametrize(
    "name",
    (
        "_comparator",
        "_comparator_symb",
        "_parallel_compare",
        "_bitonic_merge",
        "_bitonic_sort",
        "_bitonic_sort_symb",
    ),
)
def test_all_original_examples_retain_declarations(name: str) -> None:
    from qualtran.bloqs.arithmetic import sorting

    source = getattr(sorting, name).make()
    library = _library(source, 0)
    program = resolve_component_program(library, 0, sorting_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(
        native_sorting_declaration(source).signature
    )
    assert bool(program.functions[0].blocks) == isinstance(source.bitsize, int)


@pytest.mark.parametrize(
    "kind", ("Comparator", "ParallelComparators", "BitonicMerge", "BitonicSort")
)
def test_symbolic_dimensions_survive_original_wire_without_physical_body(
    kind: str,
) -> None:
    from qualtran.bloqs.arithmetic import sorting

    k, n, d = sympy.symbols("k n d", positive=True, integer=True)
    fields = {
        "Comparator": (n,),
        "ParallelComparators": (k, d, n),
        "BitonicMerge": (k, n),
        "BitonicSort": (k, n),
    }
    source = getattr(sorting, kind)(*fields[kind])
    library = _library(source, 0)
    program = resolve_component_program(library, 0, sorting_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


def test_source_nonbinary_merge_retains_declaration_and_lowering_refusal() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge

    source = BitonicMerge(3, 2)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    program = resolve_component_program(_library(source, 0), 0, sorting_factories())
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


def test_zero_pair_network_and_one_element_sort_are_physical_identities() -> None:
    from qualtran.bloqs.arithmetic.sorting import BitonicSort, ParallelComparators

    for source in (ParallelComparators(1, 2, 2), BitonicSort(1, 2)):
        program = resolve_component_program(_library(source, 0), 0, sorting_factories())
        bound = bind_register_program(program, ((0, 1),))
        assert bound.circuit.instructions == ()
        assert bound.outputs == ((0, 1),)


def test_original_zero_length_parallel_identity_remains_a_native_execution_gap() -> (
    None
):
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    source = ParallelComparators(0, 1, 2)
    assert not source.decompose_bloq().bloq_instances
    with pytest.raises(ValueError, match="positive"):
        resolve_component_program(_library(source, 0), 0, sorting_factories())


@pytest.mark.parametrize(
    "fields", ({}, {"bitsize": True}, {"bitsize": 1.0}, {"bitsize": 2, "extra": 1})
)
def test_invalid_constructor_metadata_refuses(fields: dict[str, Any]) -> None:
    with pytest.raises(SerializationError):
        sorting_factories()[_PREFIX + "Comparator"]("main", None, fields)


def test_supplied_signature_must_match_owned_directional_ports() -> None:
    with pytest.raises(SerializationError, match="signature"):
        sorting_factories()[_PREFIX + "BitonicMerge"](
            "main", (), {"half_length": 2, "bitsize": 2}
        )


@pytest.mark.parametrize(
    "kind", ("Comparator", "ParallelComparators", "BitonicMerge", "BitonicSort")
)
def test_zero_width_source_declaration_stays_bodyless(kind: str) -> None:
    from qualtran.bloqs.arithmetic import sorting

    fields = {
        "Comparator": (0,),
        "ParallelComparators": (2, 1, 0),
        "BitonicMerge": (1, 0),
        "BitonicSort": (2, 0),
    }
    source = getattr(sorting, kind)(*fields[kind])
    program = resolve_component_program(_library(source, 0), 0, sorting_factories())
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks


def test_symbolic_history_validation_rejects_a_changed_count() -> None:
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge

    source = BitonicMerge(sympy.Symbol("k", integer=True, positive=True), 2)
    library = _library(source, 0)
    entry = library.entries[0]
    ports = entry.declaration.registers
    assert ports is not None
    history = ports[-1]
    incorrect = Register(
        history.name, history.dtype, (history.shape_symbolic[0] + 1,), side=history.side
    )
    declaration = replace(entry.declaration, registers=(*ports[:-1], incorrect))
    altered = replace(library, entries=(replace(entry, declaration=declaration),))
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(altered, 0, sorting_factories())
