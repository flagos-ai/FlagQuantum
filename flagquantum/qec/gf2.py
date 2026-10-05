"""Gaussian elimination over GF(2) on bit-vector rows.

A binary vector is carried in one integer, where bit ``i`` is the coefficient of
column ``i``. Addition over GF(2) is then integer exclusive-or, and no array
library or field abstraction is involved, which keeps the questions asked here
small enough to read in one sitting: which rows of a set are independent, and
whether one more vector is a combination of them.

Those two are the whole of what a Calderbank-Shor-Steane code needs. A parity
check is a row, the code's stabilizer span is a row set, and every statement about
a code -- that its two check families commute, that a declared logical operator
lies outside the stabilizer span, that a logical basis pairs up non-degenerately --
is one of these two questions on rows read out of the code's own matrices.

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

__all__ = ("in_span", "rank", "reduce_rows", "reduce_vector")


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
