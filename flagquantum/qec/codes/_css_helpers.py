"""css helpers code definitions."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from numbers import Integral

from ..gf2 import in_span, reduce_rows, reduce_vector
from ..pauli import Pauli
from ._compat import _retain_module

_DISTANCE_SEARCH_WEIGHT = 3
BinaryMatrix = Sequence[Sequence[int]]


def _block_width(rows: tuple[tuple[int, ...], ...], *, name: str) -> int | None:
    """Return the number of columns ``rows`` states, or ``None`` when it has no row."""

    if not rows:
        return None
    width = len(rows[0])
    if width == 0:
        raise ValueError(
            f"{name} has {len(rows)} row(s) but no column, so it states an operator "
            "over no data qubit"
        )
    if any(len(row) != width for row in rows):
        raise ValueError(
            f"{name} rows disagree about how many data qubits the code has, and every "
            "matrix of a code is read over the same data qubits"
        )
    return width


def _matrix_rows(block: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    """Read ``block`` as a matrix of ``0``s and ``1``s, one row per operator."""

    if isinstance(block, (str, bytes)) or not isinstance(block, Sequence):
        raise TypeError(
            f"{name} must be a matrix of 0s and 1s with one row per check or "
            "logical operator, which is a sequence of sequences: read a tensor "
            "block with .tolist() first"
        )
    rows: list[tuple[int, ...]] = []
    for index, row in enumerate(block):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"{name} row {index} must be a sequence of 0s and 1s")
        values: list[int] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, Integral):
                raise TypeError(
                    f"{name} row {index} carries {entry!r}, and a parity-check or "
                    "logical-operator matrix has one 0-or-1 entry per data qubit"
                )
            value = int(entry)
            if value not in (0, 1):
                raise ValueError(
                    f"{name} row {index} carries {value}, and a parity-check or "
                    "logical-operator matrix has one 0-or-1 entry per data qubit"
                )
            values.append(value)
        rows.append(tuple(values))
    return tuple(rows)


def _mask(row: tuple[int, ...]) -> int:
    """Return one row as a bit vector over its columns."""

    bits = 0
    for position, value in enumerate(row):
        if value:
            bits |= 1 << position
    return bits


def _supports(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Return the qubit support of every row, in ascending qubit order."""

    return tuple(
        tuple(position for position, value in enumerate(row) if value) for row in rows
    )


def _odd_overlap(vector: int, row: int) -> bool:
    """Whether the two bit vectors agree on an odd number of columns."""

    return bin(vector & row).count("1") % 2 == 1


def _minimum_logical_weight(
    *,
    width: int,
    commuting_rows: tuple[int, ...],
    stabilizer_rows: tuple[int, ...],
    bound: int,
) -> int | None:
    """Return the least weight of a logical operator of one Pauli family.

    An operator of the family commutes with every check of the other family, which
    is ``commuting_rows``, and is not a combination of the checks of its own
    family, which is ``stabilizer_rows``. A bounded syndrome trellis gives the
    exact minimum for eligible self-dual codes. Otherwise qubit subsets are
    tried in increasing weight. Both searches stop at ``bound``.

    The stabilizer span is reduced to its pivots once, because every candidate is
    tested against the same span.
    """

    pivots = reduce_rows(list(stabilizer_rows))
    # For a self-orthogonal odd-width code encoding one qubit, the stabilizer
    # span is exactly the even-weight part of its kernel. An odd kernel word is
    # therefore precisely a logical operator, independent of its representative.
    if (
        commuting_rows == stabilizer_rows
        and width % 2 == 1
        and 2 * len(pivots) == width - 1
        and all(row.bit_count() % 2 == 0 for row in commuting_rows)
        and all(
            (left & right).bit_count() % 2 == 0
            for left in commuting_rows
            for right in commuting_rows
        )
    ):
        spans = tuple(
            ((row & -row).bit_length() - 1, row.bit_length() - 1)
            for row in commuting_rows
            if row
        )
        # At most 16 live checks plus logical parity: 2**17 states. Dense
        # unstructured matrices retain the existing bounded subset search.
        if (
            max(
                (
                    sum(first <= wire < last for first, last in spans)
                    for wire in range(width)
                ),
                default=0,
            )
            <= 16
        ):
            return _minimum_odd_kernel_weight(width, commuting_rows, bound)
    for weight in range(1, bound + 1):
        for positions in itertools.combinations(range(width), weight):
            vector = 0
            for position in positions:
                vector |= 1 << position
            if any(_odd_overlap(vector, row) for row in commuting_rows):
                continue
            if reduce_vector(vector, pivots):
                return weight
    return None


def _minimum_odd_kernel_weight(
    width: int, rows: tuple[int, ...], bound: int
) -> int | None:
    """Find the lightest odd kernel word by an exact syndrome trellis.

    A state carries the live check parities and total weight parity. Prefixes
    with the same state have identical possible suffixes, so only their least
    weight matters. A check must be zero as soon as its last data qubit is processed.
    """
    states = {0: 0}
    for wire in range(width):
        column = 1
        finished = 0
        for index, row in enumerate(rows, start=1):
            if (row >> wire) & 1:
                column |= 1 << index
            if row.bit_length() == wire + 1:
                finished |= 1 << index
        following: dict[int, int] = {}
        for syndrome, weight in states.items():
            if not syndrome & finished:
                following[syndrome] = min(following.get(syndrome, bound + 1), weight)
            selected = syndrome ^ column
            if weight < bound and not selected & finished:
                following[selected] = min(
                    following.get(selected, bound + 1), weight + 1
                )
        states = following
    return states.get(1)


def _refuse_non_logicals(
    *,
    rows: tuple[tuple[int, ...], ...],
    family: str,
    as_pauli: tuple[Pauli, ...],
    opposite_checks: tuple[Pauli, ...],
    opposite_family: str,
    own_masks: tuple[int, ...],
    own_family: str,
) -> None:
    """Refuse every row of one logical family that is not a logical operator."""

    for index, logical in enumerate(as_pauli):
        for check_index, check in enumerate(opposite_checks):
            if not logical.commutes_with(check):
                raise ValueError(
                    f"{family} row {index} anticommutes with {opposite_family} row "
                    f"{check_index}, so it does not preserve the code space and its "
                    "outcome cannot be read out"
                )
        if in_span(_mask(rows[index]), list(own_masks)):
            raise ValueError(
                f"{family} row {index} is a combination of the {own_family} rows, so "
                "it is a stabilizer whose outcome is fixed and the terminal readout "
                "would report a constant rather than the operator's eigenvalue"
            )


_retain_module(
    "flagquantum.qec.codes",
    _block_width,
    _matrix_rows,
    _mask,
    _supports,
    _odd_overlap,
    _minimum_logical_weight,
    _refuse_non_logicals,
)
