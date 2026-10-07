"""Gaussian elimination over GF(2) on bit-vector rows.

A binary vector is carried in one integer, where bit ``i`` is the coefficient of
column ``i``. Addition over GF(2) is then integer exclusive-or, and no array
library or field abstraction is involved, which keeps the questions asked here
small enough to read in one sitting: which rows of a set are independent, whether
one more vector is a combination of them, and what the rows together leave at
zero.

Those two are the whole of what a Calderbank-Shor-Steane code needs. A parity
check is a row, the code's stabilizer span is a row set, and every statement about
a code -- that its two check families commute, that a declared logical operator
lies outside the stabilizer span, that a logical basis pairs up non-degenerately --
is one of these two questions on rows read out of the code's own matrices.

**A third question is asked by the route that derives a family rather than
declaring one.** A code written down from a rule states its checks and no logical
operators, because a logical operator of a lattice or of a pair of polynomials is
not something the rule's parameters name; it is read back out of the matrices. The
Z-type logical operators of a Calderbank-Shor-Steane code are the X-check
constraints' own null space modulo the Z-type stabilizer span, and the X-type ones
are the mirror of that, so :func:`nullspace` returns one basis of the first half of
that reading. It is stated here rather than beside the family that needs it because
it is the same elimination over the same rows: a basis of the null space is what the
elimination's free columns already are, and a second home for it would be a second
Gaussian elimination to keep in step with this one.

**The reduced rows are returned rather than kept.** Reducing a row set is the
expensive half of every question, and a caller that asks the same question of many
vectors -- a minimum-weight search tries one subset after another against one fixed
span -- would pay for it once per vector if the only entry point reduced the rows
again each time. Reducing a row set to its pivots and then reducing one vector
against those pivots separates the two, so the expensive half happens once.

This module holds no state, builds no circuit, and knows nothing about codes. Its
functions are private to the package: they are an implementation detail of the code
records and of the logical-operator certification that shares them.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ("in_span", "nullspace", "rank", "reduce_rows", "reduce_vector")


def reduce_rows(rows: list[int]) -> tuple[int, ...]:
    """Return a reduced pivot for each independent row of ``rows``.

    A row that is already a combination of the pivots found so far reduces to zero
    and contributes nothing. Each row is reduced against every pivot.

    The pivots are one basis of the rows' span rather than a canonical one, so
    which pivots come back depends on the order the rows arrive in. How many come
    back does not: that number is the rank, and every combination of the rows
    reduces to nothing against it.

    **The order the pivots are returned in is load-bearing, and it is the order
    they were found.** A pivot is reduced against every pivot already kept before
    it is kept, so it carries a zero at each of their leading positions, and a pivot
    therefore never has the leading position of anything kept before it. That is
    what makes one pass of :func:`reduce_vector` enough: clearing a position is
    permanent, because no pivot left to process can set that bit again, and no pivot
    already processed is disturbed by the ones after it. Sorting the pivots would
    break exactly that: a basis in a different order no more has this property, and
    one pass over it can stop at a nonzero vector that is in the span.
    """

    pivots: list[int] = []
    for row in rows:
        current = reduce_vector(row, pivots)
        if current:
            pivots.append(current)
    return tuple(pivots)


def reduce_vector(vector: int, pivots: Sequence[int]) -> int:
    """Return what is left of ``vector`` after the pivots are subtracted from it.

    Zero means ``vector`` is a combination of the rows the pivots came from. This
    is the cheap half of :func:`in_span`, and the half a caller repeats: it takes
    one pass over the pivots rather than a fresh elimination of the whole row set.

    The pivots are expected to be a :func:`reduce_rows` result, in its own order,
    which is the only way this module produces them. One pass is enough for that
    basis and for no other: reordering it, or handing over pivots built by hand,
    gives a pass that can stop at a nonzero vector even when the vector is in the
    span. A caller with pivots of its own multiplies the rows out instead.
    """

    for pivot in pivots:
        vector = min(vector, vector ^ pivot)
    return vector


def rank(rows: list[int]) -> int:
    """Return the GF(2) rank of ``rows`` read as bit vectors."""

    return len(reduce_rows(rows))


def in_span(vector: int, rows: list[int]) -> bool:
    """Whether ``vector`` is a GF(2) combination of ``rows``."""

    return reduce_vector(vector, reduce_rows(rows)) == 0


def nullspace(rows: list[int], width: int) -> tuple[int, ...]:
    """Return one basis of the vectors ``rows`` leave at zero, over ``width`` bits.

    The basis is read off the same elimination the rest of this module uses rather
    than searched for. A column that no pivot leads is free, and the basis vector
    for a free column carries that column together with the leading column of every
    pivot whose own entry in the free column is one. That vector is orthogonal to
    every row: a row with a one in the free column meets the free column's own one
    and its own leading column's one in that same vector, and a row with a zero
    there meets nothing else, because every pivot carries zeros at each other's
    leading columns. Those columns are independent by construction, since each basis
    vector is the only one carrying its own free column, so the basis has one vector
    per free column and that count is ``width`` minus the rank.

    **Each leading column has to be made to carry exactly one, and
    :func:`reduce_rows` does not leave it that way.** That function zeroes a pivot
    at the leading columns of the pivots kept before it and says nothing about the
    ones kept after it, so a leading column can still carry a second one. The
    reading above needs the stronger property, so the pivots are put through one
    more pass, from the last kept to the first, that clears each leading column out
    of the pivots above it; a pivot is only ever added to a pivot whose leading
    column is above its own, so no leading column moves while this happens. The
    pass is what makes the basis a basis rather than a set of vectors that merely
    lie in the null space.

    **The width is a parameter because no row need state it.** The null space of no
    rows is the whole space, and a caller that asks it of an empty row set is asking
    for one basis vector per column, which is a question about a width that no row
    carries. It is also the whole of what the caller asked about: a row is read
    within the width, so a bit at or above it is outside the question and is masked
    away before anything else. Leave that masking out and a row with such a bit is
    worse than ignored -- its leading column is outside the width, so that column is
    counted as free and the basis vector built for it carries the out-of-range bit
    back out. What is returned is therefore a basis of the null space over the
    ``width`` columns, and its dimension is ``width`` minus the rank of the rows read
    within it. A width below one is refused, because the shift the masking is built
    from is not defined for it.
    """

    if width < 1:
        raise ValueError(f"the width is {width}, and a column count is at least one")
    rows = [row & ((1 << width) - 1) for row in rows]
    pivots = list(reduce_rows(rows))
    for index in range(len(pivots) - 1, -1, -1):
        leading = pivots[index].bit_length() - 1
        for earlier in range(index):
            if pivots[earlier] >> leading & 1:
                pivots[earlier] ^= pivots[index]
    occupied = {pivot.bit_length() - 1 for pivot in pivots}
    basis: list[int] = []
    for column in range(width):
        if column in occupied:
            continue
        vector = 1 << column
        for pivot in pivots:
            if pivot >> column & 1:
                vector |= 1 << (pivot.bit_length() - 1)
        basis.append(vector)
    return tuple(basis)
