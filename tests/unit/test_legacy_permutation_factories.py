"""Constructor resolution reuses complete cycle execution and source boundaries."""

from collections import Counter
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_permutation_factories import (
    permutation_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_permutation_declaration import native_declaration

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
_PREFIX = "qualtran.bloqs.arithmetic.permutation."


@pytest.mark.parametrize("composed", (False, True))
@pytest.mark.parametrize("cycles", (((0, 1),), ((0, 2), (1, 3)), ((2,),)))
def test_original_wire_resolves_to_certified_program_and_resources(
    composed: bool, cycles: tuple[tuple[int, ...], ...]
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = Permutation(4, cycles) if composed else PermutationCycle(4, cycles[0])
    library = _library(source, 0)
    program = resolve_component_program(library, 0, permutation_factories())
    assert verify_program_record(program) == ()
    expected = native_declaration(source).program()
    assert program == _rename_program(expected, program.functions[0].name)
    bound = bind_register_program(program, ((0, 1),), clean=(2, 3))
    direct = bind_register_program(expected, ((0, 1),), clean=(2, 3))
    assert bound == direct
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement


@pytest.mark.parametrize("composed", (False, True))
def test_symbolic_size_retains_original_wire_declaration(composed: bool) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle

    size = sympy.Symbol("N")
    source = (
        Permutation(size, ((0, 1),)) if composed else PermutationCycle(size, (0, 1))
    )
    library = _library(source, 0)
    program = resolve_component_program(library, 0, permutation_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == library.entries[0].declaration.registers
    assert not program.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("cycles", ((), ((0, 1), (2, 3, 4))))
def test_direct_literal_constructor_preserves_empty_identity_and_ragged_cycles(
    cycles: tuple[tuple[int, ...], ...],
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    source = Permutation(8, cycles)
    actual = permutation_factories()[_PREFIX + "Permutation"](
        "main", None, {"N": 8, "cycles": cycles}
    )
    assert actual == _rename_program(native_declaration(source).program(), "main")


@pytest.mark.parametrize(
    "name",
    (
        "_permutation_cycle_symb",
        "_permutation_symb",
        "_permutation_symb_with_cycles",
        "_sparse_permutation",
        "_sparse_permutation_with_symbolic_N",
    ),
)
def test_original_shape_and_ragged_wire_failures_remain_explicit(name: str) -> None:
    from qualtran.bloqs.arithmetic import permutation

    source = getattr(permutation, name).make()
    with pytest.raises(ValueError):
        _library(source, 0)
    # These declarations already belong to the shared sorting domain.
    assert tuple(native_declaration(source).signature)


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"N": 4, "cycle": (0, 1), "extra": 1},
        {"N": 4, "cycle": np.array([[0, 1]])},
        {"N": 4, "cycle": (0.0, 1.0)},
        {"N": 4, "cycle": (True, 1)},
    ),
)
def test_invalid_literal_cycle_metadata_refuses(arguments: dict[str, Any]) -> None:
    with pytest.raises(SerializationError):
        permutation_factories()[_PREFIX + "PermutationCycle"]("main", None, arguments)


def test_original_empty_outer_wire_and_invalid_signature() -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    library = _library(Permutation(4, ()), 0)
    program = resolve_component_program(library, 0, permutation_factories())
    assert bind_register_program(program, ((0, 1),)).circuit.instructions == ()
    with pytest.raises(SerializationError, match="signature"):
        permutation_factories()[_PREFIX + "PermutationCycle"](
            "main", (), {"N": 4, "cycle": (0, 1)}
        )
