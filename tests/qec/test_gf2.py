"""Unit coverage for the GF(2) row reduction the code records share.

The claims asserted here are the helper's own: which rows are kept, what the
reduction answers, and the form of the basis that makes one pass enough. They are
not claims about any code, which is what ``test_css_code.py`` covers, and not
about a circuit, which is what the integration files cover.

What is asserted: the pivot count is the rank rather than the row count, a row
that is a combination of the earlier ones contributes nothing, each pivot is clear
at the leading position of every pivot kept before it, and the reduction answers
the span question on that basis and not on a reordering of it. What is not
asserted: that the basis is the canonical one, because it is not — which pivots
come back depends on the arrival order, and no test here pins a particular basis
for a row set with more than one independent row.
"""

from __future__ import annotations

import itertools

import pytest

from flagquantum.qec.gf2 import (
    in_span,
    nullspace,
    rank,
    reduce_rows,
    reduce_vector,
)

pytestmark = pytest.mark.unit

_ROWS = [0b110100, 0b011010, 0b101110, 0b000011]
_DEPENDENT = [0b011, 0b101, 0b110]


def _leading(pivot: int) -> int:
    """The column a pivot's highest set bit sits in."""

    return pivot.bit_length() - 1


def test_a_row_that_is_a_combination_of_the_earlier_ones_contributes_no_pivot() -> None:
    """The third row is the exclusive-or of the first two, so it adds no basis row."""

    assert reduce_rows(_DEPENDENT[:2]) == reduce_rows(_DEPENDENT)


def test_the_rank_is_the_pivot_count_and_not_the_row_count() -> None:
    """Three rows that span two dimensions have rank two."""

    assert len(_DEPENDENT) == 3
    assert rank(_DEPENDENT) == 2
    assert rank(_DEPENDENT) == len(reduce_rows(_DEPENDENT)) < len(_DEPENDENT)


def test_the_rank_does_not_depend_on_the_order_of_the_rows() -> None:
    """Every permutation of a row set spans the same space."""

    for permutation in itertools.permutations(_DEPENDENT):
        assert rank(list(permutation)) == rank(_DEPENDENT)


def test_a_repeated_row_is_kept_once() -> None:
    """A second copy of a row is a combination of the first, so it is not a pivot."""

    assert reduce_rows([0b1011, 0b1011]) == reduce_rows([0b1011])


def test_a_pivot_is_clear_at_the_leading_position_of_every_pivot_before_it() -> None:
    """This is the invariant the one-pass reduction rests on.

    A pivot is reduced against the pivots already kept before it is kept, so it
    carries a zero at each of their leading positions. Asserted as the property
    itself rather than as a particular basis: which pivots come back depends on the
    arrival order, but this clearance does not.
    """

    pivots = reduce_rows(_ROWS)

    assert len(pivots) > 1
    assert len({_leading(pivot) for pivot in pivots}) == len(pivots)
    for index, pivot in enumerate(pivots):
        for earlier in pivots[:index]:
            assert not pivot >> _leading(earlier) & 1


def test_one_pass_is_enough_for_the_basis_the_reduction_builds() -> None:
    """A combination of the rows reduces to nothing on the basis as it comes back.

    The pivot list below is in ascending leading position, which is not the
    descending order a hand-written basis would use, so this is not a statement
    about a tidy basis: it is a statement that the arrival order is the correct
    one for the pass.
    """

    pivots = reduce_rows([0b01, 0b10])

    assert [_leading(pivot) for pivot in pivots] != [
        _leading(pivot) for pivot in reversed(sorted(pivots))
    ]
    assert reduce_vector(0b11, pivots) == 0
    assert reduce_vector(0b10, pivots) == 0


def test_the_pivots_are_returned_in_the_order_they_were_found() -> None:
    """No sort is applied to the basis on the way out.

    Pinned as evidence rather than as a claim: the reduction's answer is about the
    arrival order, and the statement here is only that the record reports the basis
    it built.
    """

    assert reduce_rows([0b10, 0b01]) == (0b10, 0b01)
    assert reduce_rows([0b01, 0b10]) == (0b01, 0b10)


def test_a_reordered_basis_is_not_reduced_correctly_by_one_pass() -> None:
    """The order is load-bearing, which is why the module does not sort.

    ``0b10`` is the exclusive-or of the two rows and therefore in their span, yet a
    single pass over the pivots in the reverse of the order they were found stops
    at ``0b01`` rather than at zero. This is the reason ``reduce_vector`` documents
    its pivots as a ``reduce_rows`` result in that result's own order.
    """

    pivots = reduce_rows([0b11, 0b01])

    assert reduce_vector(0b10, pivots) == 0
    assert reduce_vector(0b10, tuple(reversed(pivots))) != 0


def test_a_row_is_a_combination_of_the_rows_it_reduces_against() -> None:
    """Every row of a set, and every exclusive-or of them, reduces to nothing."""

    for size in (1, 2, 3):
        for combination in itertools.combinations(_ROWS, size):
            vector = 0
            for row in combination:
                vector ^= row
            assert in_span(vector, _ROWS)


def test_a_vector_outside_the_span_does_not_reduce_to_nothing() -> None:
    """The reduction answers the span question rather than always succeeding."""

    assert in_span(0b1111, [0b0011, 0b1100])
    assert not in_span(0b0001, [0b0011, 0b1100])
    assert reduce_vector(0b0001, reduce_rows([0b0011, 0b1100])) != 0


def test_reducing_against_no_pivots_is_the_vector_itself() -> None:
    """An empty basis spans only the zero vector."""

    assert reduce_rows([]) == ()
    assert reduce_vector(0b101, ()) == 0b101
    assert in_span(0, [])
    assert not in_span(1, [])


def test_the_bit_is_the_column() -> None:
    """A vector is carried as one integer and a column is its bit position.

    There is no width parameter, so the helper cannot disagree with a caller about
    how many columns a code has: a row of a hundred columns is reduced against a row
    of two by the same arithmetic.
    """

    assert in_span(1 << 40, [(1 << 40) | 1, 1])
    assert not in_span(1 << 40, [1 << 41, 1])
    assert reduce_vector(1 << 41, reduce_rows([(1 << 40) | (1 << 41)])) == 1 << 40


_SOLVABLE = (
    ([], 4),
    ([0b01, 0b10], 2),
    ([0b011], 3),
    ([0b011, 0b011, 0b100], 3),
    ([0b01], 4),
    ([0b110, 0b011, 0b101], 3),
    ([0b100, 0b011], 3),
    (_ROWS, 6),
)


@pytest.mark.parametrize(("rows", "width"), _SOLVABLE)
def test_the_null_space_dimension_is_the_width_less_the_rank(
    rows: list[int], width: int
) -> None:
    """The width is a parameter here, which is the one place this module needs one.

    A span question is answered by the rows alone: a row of a hundred columns is
    reduced against a row of two by the same arithmetic. The dimension of the null
    space is not, because it is a count of columns and no row need carry the last
    one -- the null space of no rows is the whole space of whatever width the caller
    had in mind. The rank of the same rows is measured alongside, so the dimension is
    checked against an independent elimination rather than against the arithmetic of
    this one.
    """

    basis = nullspace(rows, width)

    assert len(basis) == width - rank(rows)
    assert rank(list(basis)) == len(basis)


@pytest.mark.parametrize(("rows", "width"), _SOLVABLE)
def test_every_null_space_basis_vector_is_orthogonal_to_every_row(
    rows: list[int], width: int
) -> None:
    """Being in the null space is the property, and it is checked bit by bit.

    The dimension test above would pass for a basis of the wrong subspace of the
    right size, so orthogonality is asserted separately and against the rows as they
    arrived rather than against the reduction's pivots.
    """

    for vector in nullspace(rows, width):
        for row in rows:
            assert bin(vector & row).count("1") % 2 == 0


def test_the_null_space_of_no_rows_is_the_whole_space() -> None:
    """Every column is free when there is nothing to reduce against."""

    assert nullspace([], 4) == (1, 2, 4, 8)
    assert nullspace([0, 0], 3) == (1, 2, 4)


def test_a_row_is_read_within_the_width_and_not_beyond_it() -> None:
    """A bit above the width is outside the question rather than an extra column.

    This is the case the masking exists for. Without it the row's leading column
    lies above the width, that column is then counted as free, and the basis vector
    built for it carries the out-of-range bit back to the caller -- a basis that is
    neither inside the width nor a basis of the null space over it. Masking the row
    first makes both halves of the reading true at once: the answer stays inside the
    width, and the dimension is the width less the rank of the row as it is read.
    """

    row = (1 << 41) | 1

    assert nullspace([row], 3) == (0b010, 0b100)
    assert all(vector >> 3 == 0 for vector in nullspace([row, 1 << 40], 3))
    assert len(nullspace([row, 1 << 40], 3)) == 3 - rank([1])


@pytest.mark.parametrize("width", (0, -1))
def test_a_width_that_names_no_column_is_refused(width: int) -> None:
    """The masking shift is not defined below one, so the count is refused."""

    with pytest.raises(ValueError, match="at least one"):
        nullspace([1], width)
