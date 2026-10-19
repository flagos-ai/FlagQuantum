"""Derive a Calderbank-Shor-Steane code's logical operators from its checks.

A code stated by its two parity-check matrices is complete as a stabilizer group
and incomplete as a code: the checks say which operators are measured and say
nothing about which operators carry information. This module closes that gap the
only way it can be closed -- by computing the logical operators out of the
matrices rather than asking the caller for them -- so that an arbitrary
parity-check pair reaches the same record a declared family reaches, with the
same distance search behind it.

The computation is the quotient a Calderbank-Shor-Steane code is defined by. The
Z-type logical operators are those that commute with every X-type check and do
not lie in the span of the Z-type checks, so they are a complement of that span
inside that null space, and the X-type family is the mirror image. Nothing about
the derivation names a lattice, a family, or a generator: a family whose own rule
already fixes its logical operators, such as the bivariate-bicycle family in
:mod:`flagquantum.qec.bicycle`, arrives at the same two helpers and gets the same
answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from .codes import (
    _DISTANCE_SEARCH_WEIGHT,
    BinaryMatrix,
    CssCode,
    _block_width,
    _matrix_rows,
)
from .gf2 import in_span, nullspace, reduce_rows

__all__ = ("qldpc_code",)


def _rows(vectors: Sequence[int], width: int) -> tuple[tuple[int, ...], ...]:
    """Return bit masks as the 0-or-1 rows a code record is stated in."""

    return tuple(
        tuple(vector >> index & 1 for index in range(width)) for vector in vectors
    )


def _masks(rows: tuple[tuple[int, ...], ...]) -> list[int]:
    """Return 0-or-1 rows as bit masks, one bit per column."""

    return [sum(bit << index for index, bit in enumerate(row) if bit) for row in rows]


def _logical_basis(
    commuting: Sequence[int], stabilizers: Sequence[int], width: int
) -> list[int]:
    """Return one operator per logical qubit of one family, as bit masks.

    The answer is a basis of the quotient ``ker(commuting) / span(stabilizers)``,
    and it is built by writing that quotient down rather than by reading it off
    residues. The stabilizer span is a subspace of the null space, because the two
    check families commute, so the stabilizer basis is extended to a basis of the
    null space one independent vector at a time; the vectors that were added are a
    complement of the stabilizer span inside it, and a complement is exactly one
    operator per logical class. The count that comes back is therefore the null
    space's dimension minus the stabilizer rank, which is the number of logical
    qubits the matrices leave.

    Two steps are worth naming because a shorter statement of them is wrong. The
    extension starts from the stabilizer basis rather than from the null space's,
    so that the counted-out vectors are the ones that were *added*: starting from
    the null space's basis instead would return a set that generates the null space
    and no way to say which of its vectors carried logical information. And the
    independence test is over the running basis as a whole rather than over the
    null space's basis alone, because a candidate that the null space's basis
    already spans may still be the first vector that leaves the stabilizer span.
    """

    pivots = reduce_rows(list(stabilizers))
    kept = list(pivots)
    added: list[int] = []
    for candidate in nullspace(list(commuting), width):
        if in_span(candidate, kept):
            continue
        kept.append(candidate)
        added.append(candidate)
    return added


def _paired_basis(
    z_logicals: list[int], x_logicals: list[int]
) -> tuple[list[int], list[int]]:
    """Return the two families in the bases whose pairing is the identity.

    The pairing matrix is read one row per Z-type operator, with bit ``column`` set
    when that operator anticommutes with the X-type operator at that column. The
    elimination is then the usual one, stated on the pairing rather than on the
    operators: a row whose diagonal is zero swaps X-type operators until it is one,
    and every other row that pairs with the pivot row has the pivot's Z-type
    operator multiplied into it until it does not. A row with no one on or after its
    own column would be a row of a singular pairing, which a Calderbank-Shor-Steane
    code's two families never are; that case is refused rather than returned.

    A basis is not unique and this function does not claim one: two runs over the
    same matrices return the same answer because the elimination is deterministic,
    while a different but equally valid complement of the stabilizer span would
    return different operators that pair the same way.
    """

    size = len(z_logicals)
    pairing = [
        sum(
            0 if bin(z_logical & x_logical).count("1") % 2 == 0 else 1 << column
            for column, x_logical in enumerate(x_logicals)
        )
        for z_logical in z_logicals
    ]
    for index in range(size):
        target = next(
            (column for column in range(index, size) if pairing[index] >> column & 1),
            None,
        )
        if target is None:
            raise ValueError(
                "the derived Z-type and X-type logical operators do not pair up "
                "non-degenerately, which the two families of a Calderbank-Shor-"
                "Steane code always do: the two check matrices commute, so the "
                "quotient they define has a well-defined pairing, and a singular one "
                "means the matrices state a different code from the one they name"
            )
        if target != index:
            x_logicals[index], x_logicals[target] = (
                x_logicals[target],
                x_logicals[index],
            )
            for row in range(size):
                left = pairing[row] >> index & 1
                right = pairing[row] >> target & 1
                if left != right:
                    pairing[row] ^= (1 << index) | (1 << target)
        for row in range(size):
            if row != index and pairing[row] >> index & 1:
                z_logicals[row] ^= z_logicals[index]
                pairing[row] ^= pairing[index]
    return z_logicals, x_logicals


def qldpc_code(
    hz: BinaryMatrix,
    hx: BinaryMatrix,
    *,
    distance_search_weight: int = _DISTANCE_SEARCH_WEIGHT,
) -> CssCode:
    """Return the code an arbitrary pair of parity-check matrices states.

    ``hz`` holds the Z-type checks and ``hx`` the X-type checks, one row per check
    and one column per data qubit, in the same layout ``CssCode`` takes. The two
    logical-operator blocks are not arguments: they are derived from the checks, so
    a caller who knows a code's stabilizers and not its logical operators can still
    reach the record the declared families return, and the derivation is the
    quotient the matrices define rather than a table.

    **What is derived and what is not.** The Z-type logical operators are a
    complement of the Z-type checks' span inside the X-type checks' null space, one
    operator per logical qubit, and the X-type family is the mirror image; the two
    are then re-expressed in the bases whose anticommutation matrix is the identity,
    because a record carries a pairing and not just two lists. A basis of a quotient
    is not unique, so the operators this returns are *representatives*: an equivalent
    operator differs from one of them by a stabilizer, and the distance is not read
    off them at all but searched over the whole code, which is why a code whose
    cheapest logical operator is lighter than any declared one still reports the
    cheaper number.

    The count of logical qubits is not an argument either. It comes back as the rank
    of the matrices -- ``len(hz)`` and ``len(hx)`` do not state it, because a check
    may repeat another -- so a code whose check rows are linearly dependent reports
    the same number it would report without the repetition, and a pair of check
    families that between them span every data qubit is refused as a stabilizer
    state rather than returned as a code of no logical qubit.

    Raises:
        TypeError: If a block is not a sequence of sequences, a row is not a
            sequence, or an entry is not an integer.
        ValueError: If an entry is neither zero nor one, a block has rows but no
            column, neither block states a data qubit, or the matrices fail any of
            the invariants :class:`~flagquantum.qec.CssCode` states -- the two check
            families do not commute, the blocks disagree about the number of data
            qubits, a row acts on no data qubit, the checks leave no logical qubit,
            the derived pairing is singular, or a family has no logical operator
            within ``distance_search_weight``.

    Examples:
        >>> h = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
        >>> steane = qldpc_code(hz=h, hx=h)
        >>> steane.num_data_qubits, steane.num_ancilla_qubits, steane.distance
        (7, 6, 3)
        >>> len(steane.logical_observables)
        2
    """

    z_rows = _matrix_rows(hz, name="hz")
    x_rows = _matrix_rows(hx, name="hx")
    z_width = _block_width(z_rows, name="hz")
    x_width = _block_width(x_rows, name="hx")
    if z_width is None:
        if x_width is None:
            raise ValueError(
                "neither hz nor hx states a data qubit, so the checks do not state "
                "how many qubits the code has and no logical operator can be "
                "derived from them"
            )
        width = x_width
    else:
        width = z_width

    z_checks = _masks(z_rows)
    x_checks = _masks(x_rows)
    z_logicals = _logical_basis(x_checks, z_checks, width)
    x_logicals = _logical_basis(z_checks, x_checks, width)
    z_logicals, x_logicals = _paired_basis(z_logicals, x_logicals)
    return CssCode(
        hz=hz,
        hx=hx,
        lz=_rows(z_logicals, width),
        lx=_rows(x_logicals, width),
        distance_search_weight=distance_search_weight,
    )
