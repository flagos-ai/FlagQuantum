"""Calderbank-Shor-Steane codes given by the matrices they are written as.

A Calderbank-Shor-Steane code is a family of Z-type checks, a family of X-type
checks, and one logical operator per basis per logical qubit, all read over one
block of data qubits. The four matrices say all of that, so this record is built
from matrices and derives every count, every ancilla, and every check from them
rather than keeping a second statement that could drift from them.

**Two shapes of matrices reach this record, and both say the same thing.** The
first hands over a :class:`~flagquantum.qec.CssCodeMatrices`, whose blocks are
tensors and which states nothing about the code's distance; the second hands over
the four blocks as sequences of sequences and states the weight its distance search
reaches. The first is how a code read back out of an existing record arrives and how
a family this package derives arrives, because such a family has already searched
for its distance; the second is how a code written down by hand arrives.

**One distance reading per shape, and neither overrides the other.** A record built
from a :class:`~flagquantum.qec.CssCodeMatrices` is told its distance and checks that
the matrices can support it: a stated logical operator of weight ``w`` proves the
distance is at most ``w``, so a larger stated distance is refused. A record built
from hand-written blocks is told only how far to search and derives the distance
itself, as the least weight of a logical operator the search finds; a search that
reaches its bound without finding one is refused rather than reported. Which reading
applies is which shape the caller wrote, and a record that states both shapes or
neither is refused.

**Every invariant the record relies on is checked, because the matrices are the
caller's.** The two check families must commute, or they generate no stabilizer
group, and the checks must leave at least one logical qubit. A declared logical
operator must commute with the checks of the opposite family and must not be a
product of its own family's checks, so that it is a logical operator rather than a
stabilizer. The two logical families must pair up non-degenerately, because a code
with that many logical qubits has that many independent pairs. The pairing between
the two families is not checked beyond that rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import torch

from .codes import (
    _DISTANCE_SEARCH_WEIGHT,
    BinaryMatrix,
    CodeCheck,
    _block_width,
    _mask,
    _matrix_rows,
    _minimum_logical_weight,
    _refuse_non_logicals,
    _supports,
    ancilla_band_counts,
)
from .dem_construction import CssCodeMatrices
from .gf2 import rank
from .pauli import Pauli

__all__ = ("CssCode",)


def _row_masks(block: torch.Tensor, *, what: str) -> tuple[int, ...]:
    """Return one mask per row of ``block``, bit ``i`` standing for column ``i``."""

    masks: list[int] = []
    for index, row in enumerate(block.tolist()):
        mask = 0
        for position, value in enumerate(row):
            if value:
                mask |= 1 << position
        if mask == 0:
            raise ValueError(
                f"{what} {index} is empty, so it states the identity operator: a "
                "check and a logical observable must each act on at least one data "
                "qubit"
            )
        masks.append(mask)
    return tuple(masks)


def _support(mask: int) -> tuple[int, ...]:
    """Return the data qubits a row mask acts on, in ascending order."""

    qubits: list[int] = []
    position = 0
    remaining = mask
    while remaining:
        if remaining & 1:
            qubits.append(position)
        remaining >>= 1
        position += 1
    return tuple(qubits)


def _row_basis(rows: tuple[int, ...]) -> dict[int, int]:
    """Return a pivot map for the span of ``rows``, keyed by each pivot's top bit."""

    basis: dict[int, int] = {}
    for row in rows:
        vector = row
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector
                break
            vector ^= basis[pivot]
    return basis


def _in_span(row: int, basis: dict[int, int]) -> bool:
    """Whether ``row`` is a combination of the rows ``basis`` was built from."""

    vector = row
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            return False
        vector ^= basis[pivot]
    return True


def _stated(block: object) -> bool:
    """Whether one of the four block fields was stated rather than left at its default.

    The hand-written shape leaves an unstated block at the empty tuple it defaults
    to, and a caller who writes a block out passes a list, a tuple of rows, or a
    tensor. Reading the question as "is this the default empty tuple?" answers all
    three, which a truth test cannot: the truth value of a tensor is not defined,
    and "stated but empty" is a different mistake from "never stated at all".
    """

    return not (isinstance(block, tuple) and not block)


@dataclass(frozen=True, eq=False)
class CssCode:
    """A Calderbank-Shor-Steane code, given by the matrices it is written as.

    Columns are data qubits and every block is read over the same ones, so a data
    qubit's column index is its qubit index. Rows of ``hz`` are the Z-type checks and
    rows of ``hx`` the X-type checks; rows of ``lz`` and ``lx`` are the logical
    operators, one row per logical qubit and one column per data qubit. The checks
    are laid out Z-type first, so the Z-type checks own the lower ancilla indices.

    A record states its matrices one of two ways. A caller who holds a
    :class:`~flagquantum.qec.CssCodeMatrices` passes ``matrices`` and the ``distance``
    the code is claimed to have. A caller who writes the blocks out by hand passes
    ``hz`` and ``hx``, the logical blocks as needed, and the weight its distance
    search reaches as ``distance_search_weight``. The two readings of ``distance``,
    the validation each shape runs, and the refusal of a record that states both
    shapes or neither are stated in the module docstring.

    Args:
        matrices: The four blocks as a :class:`~flagquantum.qec.CssCodeMatrices`.
        distance: The distance the caller states, beside ``matrices``.
        hz: The Z-type checks, one row per check.
        hx: The X-type checks, one row per check.
        lz: The Z-type logical operators, one row per logical qubit.
        lx: The X-type logical operators, one row per logical qubit.
        distance_search_weight: The weight the distance search for hand-written
            blocks reaches, and the ceiling on the distance a derived shape can
            separate from its own search.

    Raises:
        TypeError: If ``matrices`` is neither ``None`` nor a
            :class:`~flagquantum.qec.CssCodeMatrices`, if a hand-written block is not
            a sequence of sequences, if a matrix entry is not an integer, or if
            ``distance`` or ``distance_search_weight`` is not an integer.
        ValueError: If a record states both shapes or neither, if a row acts on no
            data qubit, if the blocks disagree about how many data qubits the code
            has, if the two check families do not commute, if the checks leave no
            logical qubit, if a logical family declares the wrong number of rows, if
            a logical operator anticommutes with an opposite-family check or lies in
            its own family's check span, if the two families pair up degenerately, if
            the stated distance is below one or exceeds the lightest stated logical
            weight, or if a family has no logical operator within
            ``distance_search_weight``.

    Examples:
        The Steane code, whose Z-type and X-type checks are the same three weight-four
        rows and whose logical operators are the same weight-three row in each basis.

        >>> from flagquantum.qec import CssCode, SteaneCode
        >>> h = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
        >>> logical = [[1, 1, 1, 0, 0, 0, 0]]
        >>> steane = CssCode(hz=h, hx=h, lz=logical, lx=logical)
        >>> steane.num_data_qubits, steane.num_ancilla_qubits, steane.distance
        (7, 6, 3)
        >>> steane.stabilizers == SteaneCode().stabilizers
        True
    """

    matrices: CssCodeMatrices | None = None
    distance: int | None = None
    hz: BinaryMatrix = ()
    hx: BinaryMatrix = ()
    lz: BinaryMatrix = ()
    lx: BinaryMatrix = ()
    distance_search_weight: int = _DISTANCE_SEARCH_WEIGHT

    _width: int = field(init=False, repr=False)
    _z_checks: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_checks: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _z_logicals: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_logicals: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_distance: int = field(init=False, repr=False)
    _z_distance: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matrices = self.matrices
        hand_written = any(
            _stated(block) for block in (self.hz, self.hx, self.lz, self.lx)
        )
        if matrices is not None:
            if hand_written:
                raise ValueError(
                    "a CSS code record states its matrices either as a "
                    "CssCodeMatrices or as four hand-written blocks, and both were "
                    "stated: pass matrices= with distance=, or pass hz= and hx= "
                    "without matrices=, so that the record knows which of the two "
                    "readings of its distance applies"
                )
            self._read_matrices(matrices)
            return
        if not hand_written:
            raise ValueError(
                "a CSS code record states its matrices either as a CssCodeMatrices "
                "with a distance or as hand-written blocks with a "
                "distance_search_weight, and neither was stated: pass matrices= and "
                "distance=, or pass hz= and hx= (with the logical blocks as needed) "
                "and a distance_search_weight"
            )
        if self.distance is not None:
            raise ValueError(
                "a CSS code record states its distance either as the distance= of a "
                "CssCodeMatrices or as the distance_search_weight= it searches a "
                "hand-written block to, and distance= was stated beside hand-written "
                "blocks: pass matrices= with distance=, or drop distance= and let the "
                "search derive it"
            )
        self._read_hand_written_matrices()

    def _read_matrices(self, matrices: CssCodeMatrices) -> None:
        """Record the code a :class:`~flagquantum.qec.CssCodeMatrices` states.

        The caller states the distance here, so what there is to check is that the
        matrices can support it: they have to state a check, state a logical
        operator, and contain no logical operator lighter than the stated distance.
        """

        if not isinstance(matrices, CssCodeMatrices):
            raise TypeError(
                "a CSS code record is built from a CssCodeMatrices, not from "
                f"{type(matrices).__name__}"
            )
        if self.distance_search_weight != _DISTANCE_SEARCH_WEIGHT:
            raise ValueError(
                "a CSS code record built from a CssCodeMatrices states its distance "
                "directly, so a distance_search_weight= has nothing to search: pass "
                "distance=, or pass hz= and hx= and let the search derive the "
                "distance from the matrices"
            )
        stated = self.distance
        if isinstance(stated, bool) or not isinstance(stated, Integral):
            raise TypeError("CSS-code distance must be an integer")
        if stated < 1:
            raise ValueError("CSS-code distance must be at least one")
        object.__setattr__(self, "distance", int(stated))

        z_checks = _row_masks(matrices.hz, what="Z-type check")
        x_checks = _row_masks(matrices.hx, what="X-type check")
        if not z_checks and not x_checks:
            raise ValueError(
                "the matrices state no check, so they do not describe a code: a "
                "CSS code needs at least one stabilizer generator"
            )
        z_logicals = _row_masks(matrices.lz, what="Z-type logical observable")
        x_logicals = _row_masks(matrices.lx, what="X-type logical observable")
        if not z_logicals and not x_logicals:
            raise ValueError(
                "the matrices state no logical observable, so they do not describe "
                "an error-correcting code: a distance describes how many faults a "
                "logical operator survives"
            )

        self._require_logical(
            z_logicals, x_checks, _row_basis(z_checks), basis="Z", checks="X-type"
        )
        self._require_logical(
            x_logicals, z_checks, _row_basis(x_checks), basis="X", checks="Z-type"
        )
        self._require_distance_fits(int(stated), z_logicals, x_logicals)

        object.__setattr__(self, "_width", matrices.num_qubits)
        object.__setattr__(self, "_z_checks", tuple(_support(m) for m in z_checks))
        object.__setattr__(self, "_x_checks", tuple(_support(m) for m in x_checks))
        object.__setattr__(self, "_z_logicals", tuple(_support(m) for m in z_logicals))
        object.__setattr__(self, "_x_logicals", tuple(_support(m) for m in x_logicals))
        # ``CssCodeMatrices`` keeps one distance and does not split it by family, so
        # each family's distance here is that one stated distance.
        object.__setattr__(self, "_x_distance", int(stated))
        object.__setattr__(self, "_z_distance", int(stated))

    def _read_hand_written_matrices(self) -> None:
        """Record the code four hand-written blocks state, deriving its distance.

        The caller states a search weight rather than a distance, so the record has
        to derive the distance, and every invariant the derivation rests on is
        checked before it runs.
        """

        if isinstance(self.distance_search_weight, bool) or not isinstance(
            self.distance_search_weight, Integral
        ):
            raise TypeError("distance search weight must be an integer")
        if self.distance_search_weight < 1:
            raise ValueError("distance search weight must be at least one")

        blocks = {
            "hz": _matrix_rows(self.hz, name="hz"),
            "hx": _matrix_rows(self.hx, name="hx"),
            "lz": _matrix_rows(self.lz, name="lz"),
            "lx": _matrix_rows(self.lx, name="lx"),
        }
        widths: dict[str, int] = {}
        for name, rows in blocks.items():
            for index, row in enumerate(rows):
                if not any(row):
                    raise ValueError(
                        f"{name} row {index} acts on no data qubit, so it states an "
                        "operator over nothing rather than a check or a logical "
                        "operator"
                    )
            width = _block_width(rows, name=name)
            if width is not None:
                widths[name] = width
        if not widths:
            raise ValueError(
                "no block states a check or a logical operator, so the matrices do "
                "not state how many data qubits the code has"
            )
        if len(set(widths.values())) != 1:
            stated = ", ".join(f"{name} has {width}" for name, width in widths.items())
            raise ValueError(
                "the blocks disagree about how many data qubits the code has: "
                f"{stated}"
            )
        width = next(iter(widths.values()))

        z_checks = blocks["hz"]
        x_checks = blocks["hx"]
        z_logicals = blocks["lz"]
        x_logicals = blocks["lx"]
        z_masks = tuple(_mask(row) for row in z_checks)
        x_masks = tuple(_mask(row) for row in x_checks)
        z_checks_as_pauli = tuple(Pauli(z_qubits=s) for s in _supports(z_checks))
        x_checks_as_pauli = tuple(Pauli(x_qubits=s) for s in _supports(x_checks))
        for z_index, z_check in enumerate(z_checks_as_pauli):
            for x_index, x_check in enumerate(x_checks_as_pauli):
                if not z_check.commutes_with(x_check):
                    raise ValueError(
                        f"hz row {z_index} and hx row {x_index} act on an odd number "
                        "of shared data qubits, so the two check families do not "
                        "commute and generate no stabilizer group: the matrices are "
                        "not those of a stabilizer code"
                    )

        z_rank = rank(list(z_masks))
        x_rank = rank(list(x_masks))
        logical_qubits = width - z_rank - x_rank
        if logical_qubits < 1:
            raise ValueError(
                f"the {len(z_checks)} Z-type and {len(x_checks)} X-type checks span "
                f"{z_rank + x_rank} of the {width} data qubits, so no logical qubit "
                "is left and the matrices describe a stabilizer state rather than a "
                "code with a distance"
            )

        for name, rows in (("lz", z_logicals), ("lx", x_logicals)):
            if rows and len(rows) != logical_qubits:
                raise ValueError(
                    f"{name} declares {len(rows)} logical operator(s) for a code with "
                    f"{logical_qubits} logical qubit(s): a logical family has one "
                    "operator per logical qubit, and a shorter or longer list is not "
                    "one"
                )
        z_logicals_as_pauli = tuple(Pauli(z_qubits=s) for s in _supports(z_logicals))
        x_logicals_as_pauli = tuple(Pauli(x_qubits=s) for s in _supports(x_logicals))
        _refuse_non_logicals(
            rows=z_logicals,
            family="lz",
            as_pauli=z_logicals_as_pauli,
            opposite_checks=x_checks_as_pauli,
            opposite_family="hx",
            own_masks=z_masks,
            own_family="hz",
        )
        _refuse_non_logicals(
            rows=x_logicals,
            family="lx",
            as_pauli=x_logicals_as_pauli,
            opposite_checks=z_checks_as_pauli,
            opposite_family="hz",
            own_masks=x_masks,
            own_family="hx",
        )

        if z_logicals_as_pauli and x_logicals_as_pauli:
            pairing = [
                sum(
                    0 if z_logical.commutes_with(x_logical) else 1 << column
                    for column, x_logical in enumerate(x_logicals_as_pauli)
                )
                for z_logical in z_logicals_as_pauli
            ]
            pairing_rank = rank(pairing)
            if pairing_rank != logical_qubits:
                raise ValueError(
                    "the declared Z-type and X-type logical operators pair up "
                    f"degenerately: their {logical_qubits}-by-{logical_qubits} "
                    f"pairing matrix has rank {pairing_rank}, so together they "
                    "generate a smaller logical group than a code with that many "
                    "logical qubits has"
                )

        x_distance = _minimum_logical_weight(
            width=width,
            commuting_rows=z_masks,
            stabilizer_rows=x_masks,
            bound=int(self.distance_search_weight),
        )
        z_distance = _minimum_logical_weight(
            width=width,
            commuting_rows=x_masks,
            stabilizer_rows=z_masks,
            bound=int(self.distance_search_weight),
        )
        if x_distance is None or z_distance is None:
            missing = "X-type" if x_distance is None else "Z-type"
            raise ValueError(
                f"no {missing} logical operator of weight at most "
                f"{self.distance_search_weight} exists, so this record cannot report "
                "the code's distance: build it again with a distance_search_weight "
                "the search reaches, because the record does not report a distance "
                "nothing checked"
            )

        object.__setattr__(self, "hz", z_checks)
        object.__setattr__(self, "hx", x_checks)
        object.__setattr__(self, "lz", z_logicals)
        object.__setattr__(self, "lx", x_logicals)
        object.__setattr__(self, "distance", min(x_distance, z_distance))
        object.__setattr__(self, "_width", width)
        object.__setattr__(self, "_z_checks", _supports(z_checks))
        object.__setattr__(self, "_x_checks", _supports(x_checks))
        object.__setattr__(self, "_z_logicals", _supports(z_logicals))
        object.__setattr__(self, "_x_logicals", _supports(x_logicals))
        object.__setattr__(self, "_x_distance", x_distance)
        object.__setattr__(self, "_z_distance", z_distance)

    @staticmethod
    def _require_logical(
        logicals: tuple[int, ...],
        opposite_checks: tuple[int, ...],
        own_check_basis: dict[int, int],
        *,
        basis: str,
        checks: str,
    ) -> None:
        """Hold one basis's stated logical operators to both halves of the definition.

        A ``basis``-type operator is a logical operator when every ``checks``
        operator meets it on an even number of qubits -- the two Paulis then
        commute -- and when it is not a product of the ``basis``-type checks
        themselves. The independence of the stated rows follows from the same
        reading: each row becomes one observable, so two rows spanning one logical
        operator would state that operator twice.
        """

        for index, logical in enumerate(logicals):
            for check_index, check in enumerate(opposite_checks):
                if (logical & check).bit_count() % 2:
                    raise ValueError(
                        f"{basis}-type logical observable {index} overlaps {checks} "
                        f"check {check_index} on an odd number of qubits, so the two "
                        "anticommute and the stated operator is not a logical "
                        "operator of this code"
                    )
            if _in_span(logical, own_check_basis):
                raise ValueError(
                    f"{basis}-type logical observable {index} is a product of the "
                    f"code's own {basis}-type checks, so it is a stabilizer rather "
                    "than a logical operator"
                )
        if len(_row_basis(logicals)) != len(logicals):
            raise ValueError(
                f"the {basis}-type logical observables are not independent: they "
                f"span {len(_row_basis(logicals))} logical operator(s) over "
                f"{len(logicals)} stated row(s), and each row becomes one observable"
            )

    def _require_distance_fits(
        self,
        stated: int,
        z_logicals: tuple[int, ...],
        x_logicals: tuple[int, ...],
    ) -> None:
        lightest = min(mask.bit_count() for mask in z_logicals + x_logicals)
        if stated > lightest:
            raise ValueError(
                f"the stated distance {stated} exceeds the weight {lightest} of the "
                "lightest logical operator these matrices state, and a stated "
                "logical operator proves the distance is at most its own weight: "
                "state a logical operator of that weight, or a smaller distance"
            )

    @property
    def x_distance(self) -> int:
        """The least weight of an X-type logical operator of this code."""

        return self._x_distance

    @property
    def z_distance(self) -> int:
        """The least weight of a Z-type logical operator of this code."""

        return self._z_distance

    @property
    def num_data_qubits(self) -> int:
        return self._width

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self._width))

    @property
    def num_ancilla_qubits(self) -> int:
        return len(self._z_checks) + len(self._x_checks)

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        first = self._width
        return tuple(first + index for index in range(self.num_ancilla_qubits))

    @property
    def num_ancilla_x_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[0]

    @property
    def num_ancilla_z_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[1]

    @property
    def num_x_stabilizers(self) -> int:
        return len(self._x_checks)

    @property
    def num_z_stabilizers(self) -> int:
        return len(self._z_checks)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(self._z_checks):
            ancilla = self._width + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((qubit, ancilla) for qubit in support),
                )
            )
        offset = len(checks)
        for index, support in enumerate(self._x_checks):
            ancilla = self._width + offset + index
            checks.append(
                CodeCheck(
                    index=offset + index,
                    stabilizer=Pauli(x_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((ancilla, qubit) for qubit in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return tuple(
            [Pauli(z_qubits=support) for support in self._z_logicals]
            + [Pauli(x_qubits=support) for support in self._x_logicals]
        )

    def __eq__(self, other: object) -> bool:
        """Compare two records by the distance and the matrices they were built from.

        A tensor-backed record is compared block by block, because a tensor's own
        equality is a tensor and ``bool()`` of one is not a comparison. A
        hand-written record is compared block by block as the sequences it was
        normalized to.
        """

        if not isinstance(other, CssCode):
            return NotImplemented
        if self.distance != other.distance:
            return False
        if self.matrices is not None or other.matrices is not None:
            if self.matrices is None or other.matrices is None:
                return False
            return all(
                bool(getattr(self.matrices, name).equal(getattr(other.matrices, name)))
                for name in ("hz", "hx", "lz", "lx")
            )
        return (
            self.hz == other.hz
            and self.hx == other.hx
            and self.lz == other.lz
            and self.lx == other.lx
        )

    def __hash__(self) -> int:
        """Hash a hand-written record by its content, and refuse a tensor one.

        A record whose blocks are tensors has no content hash that agrees with the
        block-by-block comparison above, so it is deliberately unhashable rather
        than quietly hashed by identity. A record whose blocks are sequences is
        immutable content and hashes by it, so equal records land in one hash
        bucket and a set of them collapses.
        """

        if self.matrices is not None:
            raise TypeError(
                "unhashable type: 'CssCode': this record's matrices are a tensor, so "
                "it has no content hash that agrees with its comparison; compare "
                "records with == or hash a record built from hand-written blocks"
            )
        return hash((self.hz, self.hx, self.lz, self.lx, self.distance))
