"""General permutation declarations retain source construction and model order."""

from collections.abc import Iterable
from itertools import permutations
from typing import Any

import pytest
import sympy

from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_cycle_program,
    permutation_program,
)
from flagquantum.compiler.registers.sorting.permutation_declaration import (
    CycleShape,
    PermutationCycleDeclaration,
    PermutationDeclaration,
)
from tests.unit.test_permutation_file_audit import _key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _shape(source: Any) -> tuple[int, ...] | CycleShape:
    from qualtran.symbolics import Shaped

    return CycleShape(source.shape) if isinstance(source, Shaped) else tuple(source)


def native_declaration(
    source: Any,
) -> PermutationCycleDeclaration | PermutationDeclaration:
    """Translate original objects at the external test boundary only."""
    from qualtran.bloqs.arithmetic.permutation import PermutationCycle
    from qualtran.symbolics import Shaped

    if isinstance(source, PermutationCycle):
        return PermutationCycleDeclaration(
            source.N, _shape(source.cycle), source.bitsize
        )
    cycles = source.cycles
    return PermutationDeclaration(
        source.N,
        (
            CycleShape(cycles.shape)
            if isinstance(cycles, Shaped)
            else tuple(_shape(cycle) for cycle in cycles)
        ),
        source.bitsize,
    )


@pytest.mark.parametrize(
    "name",
    (
        "_permutation_cycle",
        "_permutation_cycle_symb_N",
        "_permutation_cycle_symb",
        "_permutation",
        "_permutation_symb",
        "_permutation_symb_with_cycles",
        "_sparse_permutation",
        "_sparse_permutation_with_symbolic_N",
    ),
)
def test_every_original_factory_uses_general_owned_declaration(name: str) -> None:
    from qualtran.bloqs.arithmetic import permutation as source_module

    source = getattr(source_module, name).make()
    declaration = native_declaration(source)
    assert declaration.size == source.N and declaration.width == source.bitsize
    assert [
        (
            r.name,
            r.bitsize,
            r.shape_symbolic,
            r.side.name,
            dict(r.dtype.parameters)["bound"],
        )
        for r in declaration.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name, r.dtype.iteration_length)
        for r in source.signature
    ]
    _, leaves = source.call_graph(max_depth=1)
    assert declaration.direct_calls() == {
        _key(child): count for child, count in leaves.items()
    }
    if isinstance(declaration, PermutationDeclaration):
        assert declaration.is_symbolic() == source.is_symbolic()
    if isinstance(source.N, int):
        assert declaration.program() == (
            permutation_cycle_program(source.N, source.cycle, source.bitsize)
            if hasattr(source, "cycle")
            else permutation_program(source.N, source.cycles, source.bitsize)
        )
    else:
        with pytest.raises(ValueError, match="concrete"):
            declaration.program()


@pytest.mark.parametrize("size", range(2, 6))
def test_complete_dense_factory_preserves_cycle_order_and_program(size: int) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    for values in permutations(range(size)):
        source = Permutation.from_dense_permutation(values)
        actual = PermutationDeclaration.from_dense(values)
        assert actual.cycles == source.cycles and actual.width == source.bitsize
        assert actual.program() == permutation_program(
            size, source.cycles, source.bitsize
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_partial_factory_retains_original_insertion_sensitive_lost_edge(
    reverse: bool,
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    values = dict(reversed(((0, 1), (2, 0)))) if reverse else {0: 1, 2: 0}
    source = Permutation.from_partial_permutation_map(3, values)
    actual = PermutationDeclaration.from_partial(3, values)
    assert actual.cycles == source.cycles == (((2, 0, 1),) if reverse else ((0, 1),))


@pytest.mark.parametrize(
    "lengths",
    ((0, 1, 2, 4, -1), (sympy.Integer(1), sympy.Symbol("n"), 3), (2, 2, 3), ()),
)
def test_cycle_length_factory_preserves_filter_symbolic_classification_and_multiplicity(
    lengths: tuple[Any, ...],
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    n = sympy.Symbol("N", positive=True, integer=True)
    source = Permutation.from_cycle_lengths(n, iter(lengths))
    actual = PermutationDeclaration.from_cycle_lengths(n, iter(lengths))
    assert actual.cycles == tuple(CycleShape(cycle.shape) for cycle in source.cycles)
    assert actual.is_symbolic() == source.is_symbolic()
    assert actual.direct_calls() == {
        _key(child): count for child, count in source.bloq_counts().items()
    }
    with pytest.raises(ValueError, match="concrete"):
        actual.program()


@pytest.mark.parametrize("shape", ((2,), (sympy.Symbol("L"),), (2, sympy.Symbol("K"))))
def test_unknown_cycle_shapes_retain_all_dimensions_and_first_dimension_cost(
    shape: tuple[Any, ...],
) -> None:
    from qualtran.bloqs.arithmetic.permutation import PermutationCycle
    from qualtran.symbolics import Shaped

    marker = CycleShape(shape)
    declaration = PermutationCycleDeclaration(8, marker)
    source = PermutationCycle(8, Shaped(shape))
    assert declaration.cycle is marker
    _, leaves = source.call_graph(max_depth=1)
    assert declaration.direct_calls() == {
        _key(child): count for child, count in leaves.items()
    }
    with pytest.raises(ValueError, match="concrete"):
        declaration.program()


def test_converter_consumes_iterators_once_and_keeps_explicit_width() -> None:
    seen: list[int] = []

    def values() -> Iterable[int]:
        for value in (2, 0, 3):
            seen.append(value)
            yield value

    cycle = PermutationCycleDeclaration(4, values(), 5)
    assert seen == [2, 0, 3] and cycle.cycle == (2, 0, 3) and cycle.width == 5
    outer = PermutationDeclaration(4, (values() for _ in range(2)), 5)
    assert outer.cycles == ((2, 0, 3), (2, 0, 3)) and outer.width == 5
    assert seen == [2, 0, 3] * 3
    assert outer.program() == permutation_program(4, ((2, 0, 3), (2, 0, 3)), 5)


def test_empty_outer_identity_and_original_empty_cycle_model_failure_are_distinct() -> (
    None
):
    from qualtran.bloqs.arithmetic.permutation import PermutationCycle

    declaration = PermutationDeclaration(4, ())
    assert declaration.direct_calls() == {} and not declaration.is_symbolic()
    assert declaration.program() == permutation_program(4, (), 2)
    with pytest.raises(IndexError):
        PermutationCycle(4, ()).bloq_counts()
    with pytest.raises(IndexError):
        PermutationCycleDeclaration(4, ()).direct_calls()
    with pytest.raises(ValueError, match="cycle"):
        PermutationCycleDeclaration(4, ()).program()
