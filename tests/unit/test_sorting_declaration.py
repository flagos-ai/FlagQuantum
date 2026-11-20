"""Complete sorting declarations share the source models and physical owners."""

from collections.abc import Hashable
from typing import Any, TypeAlias

import pytest
import sympy

from flagquantum.compiler.registers.comparison.compare_exchange_component import (
    compare_exchange_program,
)
from flagquantum.compiler.registers.comparison.ordering_declaration import (
    ComparatorDeclaration,
    ParallelComparatorsDeclaration,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.sorting.bitonic_declaration import (
    BitonicMergeDeclaration,
    BitonicSortDeclaration,
)
from flagquantum.compiler.registers.sorting.bitonic_merge_component import (
    bitonic_merge_program,
)
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
)
from tests.unit.test_sorting_file_audit import _key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Declaration: TypeAlias = (
    ComparatorDeclaration
    | ParallelComparatorsDeclaration
    | BitonicMergeDeclaration
    | BitonicSortDeclaration
)


def native_sorting_declaration(source: Any) -> Declaration:
    """Translate original objects only at the external reference boundary."""
    from qualtran.bloqs.arithmetic.sorting import (
        BitonicMerge,
        BitonicSort,
        Comparator,
        ParallelComparators,
    )

    if isinstance(source, Comparator):
        return ComparatorDeclaration(source.bitsize)
    if isinstance(source, ParallelComparators):
        return ParallelComparatorsDeclaration(source.k, source.offset, source.bitsize)
    if isinstance(source, BitonicMerge):
        return BitonicMergeDeclaration(source.half_length, source.bitsize)
    assert isinstance(source, BitonicSort)
    return BitonicSortDeclaration(source.k, source.bitsize)


def assert_declaration(source: Any) -> Declaration:
    declaration = native_sorting_declaration(source)
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name, r.dtype.encoding)
        for r in declaration.signature
    ] == [
        (
            r.name,
            r.bitsize,
            r.shape_symbolic,
            r.side.name,
            "bit" if type(r.dtype).__name__ == "QBit" else "uint",
        )
        for r in source.signature
    ]
    expected: dict[Hashable, Any] = {
        _key(child): count for child, count in source.bloq_counts().items()
    }
    assert declaration.direct_calls() == expected
    if not isinstance(declaration, ComparatorDeclaration):
        assert declaration.is_symbolic() == source.is_symbolic()
        assert declaration.num_comparisons == source.num_comparisons
    return declaration


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
def test_every_actual_factory_uses_generic_owned_declaration(name: str) -> None:
    from qualtran.bloqs.arithmetic import sorting

    source = getattr(sorting, name).make()
    declaration = assert_declaration(source)
    if not isinstance(source.bitsize, int):
        with pytest.raises(ValueError, match="concrete"):
            declaration.program()


@pytest.mark.parametrize("count", range(1, 10))
@pytest.mark.parametrize("offset", range(1, 10))
def test_all_source_finite_offsets_retain_exact_counts_and_metadata(
    count: int, offset: int
) -> None:
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    assert_declaration(ParallelComparators(count, offset, 3))


@pytest.mark.parametrize("which", ("count", "offset", "width", "all"))
def test_generic_symbolic_parallel_uses_original_half_count_upper_bound(
    which: str,
) -> None:
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    k, d, n = sympy.symbols("k d n", positive=True, integer=True)
    source = ParallelComparators(
        k if which in ("count", "all") else 7,
        d if which in ("offset", "all") else 2,
        n if which in ("width", "all") else 3,
    )
    declaration = assert_declaration(source)
    with pytest.raises(ValueError, match="concrete"):
        declaration.program()


@pytest.mark.parametrize("kind", ("merge", "sort"))
@pytest.mark.parametrize("symbolic_length", (False, True))
@pytest.mark.parametrize("symbolic_width", (False, True))
def test_generic_merge_sort_symbolic_signatures_and_counts(
    kind: str, symbolic_length: bool, symbolic_width: bool
) -> None:
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, BitonicSort

    k, n = sympy.symbols("k n", positive=True, integer=True)
    source = (BitonicMerge if kind == "merge" else BitonicSort)(
        k if symbolic_length else 4, n if symbolic_width else 2
    )
    declaration = assert_declaration(source)
    if symbolic_length or symbolic_width:
        with pytest.raises(ValueError, match="concrete"):
            declaration.program()


@pytest.mark.parametrize("adjoint", (False, True))
def test_finite_conversion_reuses_all_existing_directional_graph_owners(
    adjoint: bool,
) -> None:
    assert ComparatorDeclaration(2).program(
        adjoint=adjoint
    ) == compare_exchange_program(2, adjoint=adjoint)
    assert ParallelComparatorsDeclaration(3, 1, 2).program(
        adjoint=adjoint
    ) == parallel_comparison_program(3, 1, 2, adjoint=adjoint)
    assert BitonicMergeDeclaration(2, 2).program(
        adjoint=adjoint
    ) == bitonic_merge_program(2, 2, adjoint=adjoint)
    assert BitonicSortDeclaration(4, 2).program(
        adjoint=adjoint
    ) == bitonic_sort_program(4, 2, adjoint=adjoint)


def test_original_invalid_lengths_and_declaration_only_nonbinary_merge() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, BitonicSort

    for length in (0, -1):
        with pytest.raises(AssertionError, match="positive"):
            BitonicMerge(length, 2)
        with pytest.raises(ValueError, match="positive"):
            BitonicMergeDeclaration(length, 2)
    with pytest.raises(AssertionError, match="power of 2"):
        BitonicSort(3, 2)
    with pytest.raises(ValueError, match="power of two"):
        BitonicSortDeclaration(3, 2)
    declaration = assert_declaration(BitonicMerge(3, 2))
    with pytest.raises(ValueError, match="power of two"):
        declaration.program()
    with pytest.raises(DecomposeNotImplementedError, match="power of 2"):
        BitonicMerge(3, 2).decompose_bloq()
