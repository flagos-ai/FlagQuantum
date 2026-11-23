"""List source constructors reuse typed declarations and literal duplicate graphs."""

from collections import Counter
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_in_place_sort_factories import (
    in_place_sort_factories,
)
from flagquantum.compiler.formats.legacy_sorted_duplicates_factories import (
    sorted_duplicates_factories,
)
from flagquantum.compiler.formats.legacy_symmetric_difference_factories import (
    symmetric_difference_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_sorted_duplicates_component import bind_case

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("length", (2, 3, 4))
def test_duplicate_constructor_program_operands_feedback_and_resources_equal_owner(
    signed: bool, controlled: bool, length: int
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = HasDuplicates(length, (QInt if signed else QUInt)(2), controlled)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, sorted_duplicates_factories())
    dtype = signed_register_type(2) if signed else unsigned_register_type(2)
    expected = sorted_duplicates_program(length, dtype, controlled=controlled)
    assert program == _rename_program(expected, program.functions[0].name)
    assert verify_program_record(program) == ()
    direct, ports, logical = bind_case(length, 2, signed, controlled)
    actual = bind_register_program(
        program, ports, clean=tuple(range(logical, direct.circuit.n_wires))
    )
    assert actual == direct
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in actual.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize("kind", ("sort", "difference", "duplicates"))
@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("signed", (False, True))
def test_bodyless_and_symbolic_source_declarations_keep_original_typed_ports(
    kind: str, symbolic: bool, signed: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import (
        HasDuplicates,
        SortInPlace,
        SymmetricDifference,
    )

    length, width = (
        sympy.symbols("l n", positive=True, integer=True) if symbolic else (3, 2)
    )
    dtype = (QInt if signed else QUInt)(width)
    source = {
        "sort": SortInPlace(length, dtype),
        "difference": SymmetricDifference(length, length, length, dtype),
        "duplicates": HasDuplicates(length if symbolic else 1, dtype),
    }[kind]
    factories = {
        **in_place_sort_factories(),
        **symmetric_difference_factories(),
        **sorted_duplicates_factories(),
    }
    library = _library(source, 0)
    program = resolve_component_program(library, 0, factories)
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize(
    "module,names",
    (
        (
            "has_duplicates",
            ("_has_duplicates", "_has_duplicates_symb", "_has_duplicates_symb_len"),
        ),
        (
            "symmetric_difference",
            ("_symm_diff", "_symm_diff_symb", "_symm_diff_equal_size_symb"),
        ),
    ),
)
def test_every_actual_example_resolves_from_untouched_wire(
    module: str, names: tuple[str, ...]
) -> None:
    from importlib import import_module

    factories = {**symmetric_difference_factories(), **sorted_duplicates_factories()}
    for name in names:
        source = getattr(
            import_module("qualtran.bloqs.arithmetic.lists." + module), name
        ).make()
        library = _library(source, 0)
        program = resolve_component_program(library, 0, factories)
        assert verify_program_record(program) == ()
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )


@pytest.mark.parametrize("controlled", (False, True))
def test_duplicate_singleton_retains_source_body_boundary(controlled: bool) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates

    program = resolve_component_program(
        _library(HasDuplicates(1, QUInt(2), controlled), 0),
        0,
        sorted_duplicates_factories(),
    )
    if controlled:
        bound = bind_register_program(program, ((0, 1), (2,), (3,)))
        assert tuple(
            (gate.name, gate.wires) for gate in bound.circuit.instructions
        ) == (("cx", (3, 2)), ("cx", (3, 2)))
    else:
        assert not program.functions[0].blocks


@pytest.mark.parametrize(
    "fields",
    (
        {},
        {"l": True, "dtype": unsigned_register_type(2)},
        {"l": 2, "dtype": "uint"},
        {"l": 2, "dtype": unsigned_register_type(2), "is_controlled": 2},
    ),
)
def test_malformed_duplicate_constructor_refuses(fields: dict[str, Any]) -> None:
    with pytest.raises(SerializationError):
        next(iter(sorted_duplicates_factories().values()))("main", None, fields)


def test_list_factories_validate_constructor_fields_and_register_signatures() -> None:
    dtype = unsigned_register_type(2)
    for factories, fields in (
        (in_place_sort_factories(), {"l": 3, "dtype": dtype}),
        (
            symmetric_difference_factories(),
            {"n_lhs": 2, "n_rhs": 1, "n_diff": 1, "dtype": dtype},
        ),
        (sorted_duplicates_factories(), {"l": 2, "dtype": dtype}),
    ):
        factory = next(iter(factories.values()))
        with pytest.raises(SerializationError, match="signature"):
            factory("main", (), fields)
        with pytest.raises(SerializationError):
            factory("main", None, fields | {"extra": 1})
