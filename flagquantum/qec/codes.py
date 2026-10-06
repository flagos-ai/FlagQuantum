"""Code-independent stabilizer-code descriptions.

A code declares its wire layout, its checks, and its logical observables. The
frozen repetition profile keeps its own records; this module describes a code as
a value so that circuit generation, detector layout, and decoding can be derived
from it rather than pinned to one instance.

Two routes reach such a record. A family declares its own layout from a distance
or an index -- the repetition profile, the Steane code, the triangular colour
patch and the square-lattice torus all do -- and :class:`CssCode` instead takes
the four matrices a Calderbank-Shor-Steane code is already written down as, which
is the route a code this package does not declare arrives by. The two surface
patches declare their layout the same way, from a lattice rather than from
matrices, and live in :mod:`flagquantum.qec.surface` because that lattice -- and
the frame conjugation cut from it -- is the machinery they share. The two routes meet:
:func:`triangular_colour_code` and :func:`toric_code` derive a family's matrices
and return the same :class:`CssCode` a caller would otherwise have written out by
hand. Every route satisfies the same protocol, so no consumer can tell which one
it was handed.

A check names one ancilla and the CNOT pairs that couple it to the stabilizer's
support, and the direction of each pair is fixed by the factor that wire carries.
A pure Z-type check therefore uses one direction alone, a pure X-type check the
other, and a mixed X-and-Z check -- which is what every check of the ZXXZ surface
patch is -- uses both directions through the same ancilla. The two surface
records are named here only by :mod:`flagquantum.qec.surface`, which is where
their lattice and that conjugation are described.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol, runtime_checkable

from .gf2 import in_span, rank, reduce_rows, reduce_vector
from .pauli import Pauli


@dataclass(frozen=True)
class CodeCheck:
    """One stabilizer check with its ancilla and its CNOT coupling.

    Each entry of ``cnot_wires`` is a ``(control, target)`` pair, and every pair
    couples the check's single ancilla to one data wire of the stabilizer's
    support. Which way a pair points is fixed by the stabilizer factor it
    carries rather than left to the caller:

    * A **Z factor** on a data wire is read by coupling that data wire into the
      ancilla with the data wire controlling, so it comes first:
      ``(data_wire, ancilla)``.
    * An **X factor** is read by conjugating the ancilla with ``H`` and coupling
      it out into that data wire with the ancilla controlling:
      ``(ancilla, data_wire)``.

    So ``cnot_wires`` is the Z-factor pairs in the stabilizer's own wire order,
    then the X-factor pairs in the stabilizer's own wire order, and the boundary
    between the two runs is what says which pairs the ``H`` gates wrap. The
    emitted gadget is the Z pairs, then ``H``, then the X pairs, then ``H``,
    then a Z-basis readout of the ancilla — which leaves that readout equal to
    the check's eigenvalue.

    **One ancilla measures a mixed stabilizer.** A pure Z-type check has an
    empty X run and no ``H`` gates at all; a pure X-type check has an empty Z
    run and is the ``H``-wrapped gadget alone. Those are the two shapes this
    record described before it described the mixed one, and they are not two
    rules beside a third: they are the two ends of the one rule above, and they
    are emitted by it unchanged.

    **Why the X run needs the H gates and the Z run does not.** Reading a Z
    factor through the ancilla is a CNOT in one direction, and reading an X
    factor is the same CNOT conjugated by ``H`` on the ancilla, because
    ``H X H = Z``. A mixed stabilizer needs both directions through the *same*
    ancilla, so it needs the ``H`` pair around the X half only — never a second
    ancilla, and never an ``H`` that would also conjugate the Z half and turn
    the check into a different operator.
    """

    index: int
    stabilizer: Pauli
    ancilla_wire: int
    cnot_wires: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("check index must be non-negative")
        if not isinstance(self.stabilizer, Pauli):
            raise TypeError("check stabilizer must be a Pauli operator")
        if self.ancilla_wire < 0:
            raise ValueError("check ancilla wire must be non-negative")
        if not self.cnot_wires:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_wires:
            if control < 0 or target < 0:
                raise ValueError("check CNOT wires must be non-negative")
        if not self.stabilizer.support:
            raise ValueError(
                "check stabilizer must have a non-empty support, because an "
                "identity check measures nothing and its ancilla has nothing to "
                "read"
            )
        self._validate_z_run()
        self._validate_x_run()

    def _validate_z_run(self) -> None:
        """Require the leading run to be the Z factors, each into the ancilla."""

        z_wires = self.stabilizer.z_wires
        run = self.cnot_wires[: len(z_wires)]
        for _, target in run:
            if target != self.ancilla_wire:
                raise ValueError(
                    "check CNOT targets an X-factor pair, which must come after "
                    "the Z-factor pairs, or couples a wire the stabilizer does "
                    "not carry a Z factor on"
                )
        controls = tuple(sorted(control for control, _ in run))
        if self.ancilla_wire in controls:
            raise ValueError(
                "check CNOT controls must be data wires, not the ancilla, "
                "because a Z-factor pair draws a data wire into the ancilla"
            )
        if controls != z_wires:
            raise ValueError(
                "check CNOT controls must match the stabilizer's Z-factor support"
            )

    def _validate_x_run(self) -> None:
        """Require the trailing run to be the X factors, each out of the ancilla."""

        x_wires = self.stabilizer.x_wires
        run = self.cnot_wires[len(self.stabilizer.z_wires) :]
        for control, _ in run:
            if control != self.ancilla_wire:
                raise ValueError(
                    "check CNOT controls must be the declared ancilla, because an "
                    "X-factor pair is controlled by the declared ancilla and its "
                    "target data wires are the stabilizer's X-factor support"
                )
        targets = tuple(sorted(target for _, target in run))
        if targets != x_wires:
            raise ValueError(
                "check CNOT targets must match the stabilizer's X-factor support"
            )


@runtime_checkable
class StabilizerCode(Protocol):
    """A code that declares its wire layout, checks, and logical observables."""

    @property
    def distance(self) -> int: ...

    @property
    def num_data_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def data_wires(self) -> tuple[int, ...]: ...

    @property
    def ancilla_wires(self) -> tuple[int, ...]: ...

    @property
    def checks(self) -> tuple[CodeCheck, ...]: ...

    @property
    def stabilizers(self) -> tuple[Pauli, ...]: ...

    @property
    def logical_observables(self) -> tuple[Pauli, ...]: ...


@dataclass(frozen=True)
class RepetitionCode:
    """The bit-flip repetition code with ``distance`` data qubits.

    Data qubits occupy wires ``0..distance-1`` and check ancillas occupy wires
    ``distance..2*distance-2``. Check ``c`` measures ``Z_c Z_{c+1}``, so the code
    detects bit flips. The single declared logical observable is ``Z`` on every
    data wire, which makes its readout representative the parity of the terminal
    data measurements.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("repetition-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("repetition-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return self.distance - 1

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self.distance))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return tuple(range(self.distance, 2 * self.distance - 1))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return tuple(
            CodeCheck(
                index=index,
                stabilizer=Pauli(z_wires=(index, index + 1)),
                ancilla_wire=self.distance + index,
                cnot_wires=(
                    (index, self.distance + index),
                    (index + 1, self.distance + index),
                ),
            )
            for index in range(self.distance - 1)
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=self.data_wires),)


_STEANE_CHECK_SUPPORTS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)
_STEANE_LOGICAL_SUPPORT: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class SteaneCode:
    """The seven-qubit Steane code, declared by its checks and its logicals.

    Data qubits occupy wires ``0..6`` and the six check ancillas follow them on
    wires ``7..12``: checks ``0..2`` are Z-type and checks ``3..5`` are X-type,
    each with the same support, which is what makes the code Calderbank-Shor-
    Steane. The supports are the three non-zero parity constraints of the
    ``[7, 4, 3]`` Hamming code, so each check has weight four and the code has
    distance three.

    Both logical operators are declared, on the same three data wires: ``Z`` on
    ``(0, 1, 2)`` and ``X`` on ``(0, 1, 2)``. They anticommute, since they overlap
    on an odd number of wires, and neither is a product of checks. This is the
    smallest code this package declares whose two fault families are both
    non-trivial, which is why it is the record the matrix route is exercised on.

    The record states the checks and the observables. It does not choose a
    decoding strategy or a noise model, and it declines to describe a
    preparation, so a memory circuit built from it is Z memory like every other
    record here.
    """

    @property
    def distance(self) -> int:
        return 3

    @property
    def num_data_qubits(self) -> int:
        return 7

    @property
    def num_ancilla_qubits(self) -> int:
        return 6

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(7))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return tuple(range(7, 13))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(_STEANE_CHECK_SUPPORTS):
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_wires=support),
                    ancilla_wire=ancilla,
                    cnot_wires=tuple((wire, ancilla) for wire in support),
                )
            )
        for offset, support in enumerate(_STEANE_CHECK_SUPPORTS):
            index = len(_STEANE_CHECK_SUPPORTS) + offset
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(x_wires=support),
                    ancilla_wire=ancilla,
                    cnot_wires=tuple((ancilla, wire) for wire in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (
            Pauli(z_wires=_STEANE_LOGICAL_SUPPORT),
            Pauli(x_wires=_STEANE_LOGICAL_SUPPORT),
        )


_DISTANCE_SEARCH_WEIGHT = 3
# The weight up to which :class:`CssCode` proves the distance it reports, and the
# default for the record's own knob.
#
# A Calderbank-Shor-Steane code has one distance per Pauli family and neither has a
# closed form, so the record looks for a logical operator of the least weight there
# is and refuses to be built when the search reaches this weight without finding
# one. Three admits every distance-three code, which is what the families declared
# here are, and it is what makes the default affordable: the search tries every wire
# subset up to the bound, so raising the bound to four costs an order of magnitude.
# Measured on the repetition profile read as a code with no low-weight logical
# operator, a bound of three takes 0.36 s at 100 data qubits and 1.3 s at 140, while
# a bound of four takes 7.2 s and 37.5 s. A code of higher distance is built by
# asking for a bound the search reaches, which is where that cost becomes the
# caller's decision rather than the default's.

BinaryMatrix = Sequence[Sequence[int]]
# The matrices a Calderbank-Shor-Steane code is written down as: one inner sequence
# per check or logical operator, holding one ``0`` or ``1`` per data qubit.


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


def _supports(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Return the wire support of every row, in ascending wire order."""

    return tuple(
        tuple(position for position, value in enumerate(row) if value) for row in rows
    )


def _mask(row: tuple[int, ...]) -> int:
    """Return one row as a bit vector over its columns."""

    bits = 0
    for position, value in enumerate(row):
        if value:
            bits |= 1 << position
    return bits


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
    family, which is ``stabilizer_rows``. Wire subsets are tried in increasing
    weight, so the first operator found has the least weight there is, and the
    search stops at ``bound``.

    The stabilizer span is reduced to its pivots once, because every candidate is
    tested against the same span.
    """

    pivots = reduce_rows(list(stabilizer_rows))
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


@dataclass(frozen=True)
class CssCode:
    """A Calderbank-Shor-Steane code given by the four matrices it is written as.

    ``hz`` holds the Z-type checks and ``hx`` the X-type checks, one row per check
    and one column per data qubit. ``lz`` and ``lx`` hold the Z-type and X-type
    logical operators in the same layout. A block with no row states that the code
    has no check or no logical operator of that family, and an omitted block states
    the same thing. Data qubits are the columns and occupy wires ``0..n-1`` in
    column order; the ancillas follow them with the Z-type checks first, so the wire
    layout is a function of the matrices rather than of the caller's numbering.

    **Every invariant the record relies on is checked here, because the matrices
    are the caller's.** The two check families must commute, or they generate no
    stabilizer group and the matrices are not those of a stabilizer code. A
    declared logical operator must commute with the checks of the opposite family
    and must lie outside the span of the checks of its own family, or it is a
    stabilizer whose outcome is fixed and which therefore carries no information. A
    declared pair of families must pair up non-degenerately, because logical
    operators whose pairing matrix is singular do not generate the logical group a
    code of that many logical qubits has. None of these is repairable: a set of
    matrices that fails one of them is not the code the caller named.

    **The distance is computed, not declared.** A Calderbank-Shor-Steane code has
    one distance per Pauli family, and :attr:`x_distance` and :attr:`z_distance`
    are those two while :attr:`distance` is the smaller of them. Each is the least
    weight of a logical operator of its family, found by trying wire subsets in
    increasing weight, so ``distance_search_weight`` states how far the search may
    look and the record refuses to be built when a family has no logical operator
    within that weight. Three is the default, which admits every distance-three
    code; a code of higher distance is built by asking for a bound the search
    reaches.

    **The repetition code is the case where the two distances differ.** Its own
    record's :attr:`distance` is the bit-flip distance it is named for, and a
    single ``Z`` on any one of its data wires commutes with all of its checks and
    lies outside their span, so its Z distance is one, which is what this record
    reports for it. The two numbers do not disagree; they are two different
    quantities, and the claim that a code has distance ``d`` is the claim about the
    smaller one.

    Raises:
        TypeError: If a block is not a sequence of sequences, a row is not a
            sequence, an entry is not an integer, or ``distance_search_weight`` is
            not an integer.
        ValueError: If an entry is neither zero nor one, a block has rows but no
            column, a row acts on no data qubit, the blocks disagree about how many
            data qubits the code has, no block states that number at all, the two
            check families do not commute, the checks leave no logical qubit, a
            logical family has the wrong number of operators, a logical operator
            anticommutes with a check of the opposite family or lies in its own
            family's stabilizer span, the pairing matrix is singular, or a family
            has no logical operator within ``distance_search_weight``.

    Examples:
        >>> h = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
        >>> logical = [[1, 1, 1, 0, 0, 0, 0]]
        >>> steane = CssCode(hz=h, hx=h, lz=logical, lx=logical)
        >>> steane.num_data_qubits, steane.num_ancilla_qubits, steane.distance
        (7, 6, 3)
        >>> steane.stabilizers == SteaneCode().stabilizers
        True
    """

    hz: BinaryMatrix
    hx: BinaryMatrix
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
        z_checks_as_pauli = tuple(Pauli(z_wires=s) for s in _supports(z_checks))
        x_checks_as_pauli = tuple(Pauli(x_wires=s) for s in _supports(x_checks))
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
        z_logicals_as_pauli = tuple(Pauli(z_wires=s) for s in _supports(z_logicals))
        x_logicals_as_pauli = tuple(Pauli(x_wires=s) for s in _supports(x_logicals))
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
        for family, found in (("X-type", x_distance), ("Z-type", z_distance)):
            if found is None:
                raise ValueError(
                    f"no {family} logical operator of weight at most "
                    f"{self.distance_search_weight} exists, so this record cannot "
                    "report the code's distance: build it again with a "
                    "distance_search_weight the search reaches, because the record "
                    "does not report a distance nothing checked"
                )

        object.__setattr__(self, "hz", z_checks)
        object.__setattr__(self, "hx", x_checks)
        object.__setattr__(self, "lz", z_logicals)
        object.__setattr__(self, "lx", x_logicals)
        object.__setattr__(self, "_width", width)
        object.__setattr__(self, "_z_checks", z_checks)
        object.__setattr__(self, "_x_checks", x_checks)
        object.__setattr__(self, "_z_logicals", z_logicals)
        object.__setattr__(self, "_x_logicals", x_logicals)
        object.__setattr__(self, "_x_distance", x_distance)
        object.__setattr__(self, "_z_distance", z_distance)

    @property
    def distance(self) -> int:
        """The code's distance, which is the smaller of its two family distances."""

        return min(self._x_distance, self._z_distance)

    @property
    def x_distance(self) -> int:
        """The least weight of an X-type logical operator of the code."""

        return self._x_distance

    @property
    def z_distance(self) -> int:
        """The least weight of a Z-type logical operator of the code."""

        return self._z_distance

    @property
    def num_data_qubits(self) -> int:
        return self._width

    @property
    def num_ancilla_qubits(self) -> int:
        return len(self._z_checks) + len(self._x_checks)

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self._width))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return tuple(range(self._width, self._width + self.num_ancilla_qubits))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(_supports(self._z_checks)):
            ancilla = self._width + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_wires=support),
                    ancilla_wire=ancilla,
                    cnot_wires=tuple((wire, ancilla) for wire in support),
                )
            )
        for offset, support in enumerate(_supports(self._x_checks)):
            index = len(self._z_checks) + offset
            ancilla = self._width + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(x_wires=support),
                    ancilla_wire=ancilla,
                    cnot_wires=tuple((ancilla, wire) for wire in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        z_logicals = tuple(Pauli(z_wires=s) for s in _supports(self._z_logicals))
        x_logicals = tuple(Pauli(x_wires=s) for s in _supports(self._x_logicals))
        return z_logicals + x_logicals


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


def triangular_colour_code(distance: int) -> CssCode:
    """Return the triangular 6.6.6 colour code of the requested distance.

    A colour code is written down on a trivalent lattice whose faces are
    three-colourable, and every face carries one X-type and one Z-type stabilizer
    over the data qubits at its vertices. That is what makes the family self-dual,
    which is why the two check blocks below are one matrix, and it is also why the
    family is not a matching code: a face in the bulk has six vertices, so one data
    fault inside it can flip more detectors than an edge can join.

    **The patch is a triangle and its side is the distance.** A point ``(row,
    column)`` with ``0 <= column <= row <= bound`` is either a data vertex or a face
    centre, and the face centres are the points where ``column % 3 == 2 - row % 3``.
    Every other point is a data vertex. A face is the six in-patch points one step
    diagonally or orthogonally away from it, so a face in the bulk has weight six
    and a face on the triangle's edge has weight four, where the boundary cuts the
    hexagon away. The declared logical operator is the data vertices on the
    ``column == 0`` side, which is the patch's shortest logical string; it commutes
    with every face and overlaps itself on an odd number of vertices, so it is a
    logical operator of both families rather than of one.

    **The requested distance is proved rather than stated.** The record handed back
    is a :class:`CssCode`, so ``distance``, ``x_distance`` and ``z_distance`` are
    searched for over the matrices derived here. The bound that search is given is
    the requested distance itself, since a smaller one would refuse the code whose
    distance is the number asked for, and that bound is the whole cost of this
    function: the search tries every wire subset up to the distance, so the family
    is affordable at the small distances an experiment is built from and is not a
    way to write down a large patch.

    **At distance three this is the Steane code, relabelled.** The smallest patch
    has seven data vertices over three faces, and its three weight-four checks are
    the Steane code's three weight-four checks under a permutation of the data
    wires -- the identity is not that permutation, which
    ``tests/qec/test_colour_code.py`` exhibits rather than asserts. The two records
    are therefore one code in two descriptions at distance three rather than two
    seven-qubit codes, and their wire numbers are not comparable position by
    position. Nothing here reads that equivalence, because the distance is proved
    from this patch's own matrices, and the family grows away from it: at distance
    five the patch has nineteen data vertices and checks of weight four and six,
    which no Steane-style description reaches.

    Args:
        distance: The code distance, an odd integer of at least three.

    Returns:
        The code record, with one logical qubit and both family distances equal to
        ``distance``.

    Raises:
        TypeError: If ``distance`` is not an integer.
        ValueError: If ``distance`` is even or below three, or if no logical
            operator of a family reaches the requested weight.

    Examples:
        >>> colour = triangular_colour_code(3)
        >>> colour.num_data_qubits, colour.num_ancilla_qubits, colour.distance
        (7, 6, 3)
        >>> colour.stabilizers[0] == Pauli(z_wires=(0, 1, 2, 3))
        True
    """

    if isinstance(distance, bool) or not isinstance(distance, Integral):
        raise TypeError("colour-code distance must be an integer")
    if distance % 2 == 0:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no whole side: "
            "the side spans (distance - 1) lattice periods of three rows, so the "
            "distance must be odd"
        )
    if distance < 3:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no face and "
            "therefore no check, so it is not a code"
        )

    bound = 3 * (distance - 1) // 2
    points = tuple(
        (row, column) for row in range(bound + 1) for column in range(row + 1)
    )
    # A face centre is the point whose column sits two places back from its row in
    # the three-colouring, in the row coordinates the patch is written in.
    faces = tuple(point for point in points if point[1] % 3 == 2 - point[0] % 3)
    vertices = tuple(point for point in points if point[1] % 3 != 2 - point[0] % 3)
    wire_of = {vertex: index for index, vertex in enumerate(vertices)}

    offsets = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1))
    supports = tuple(
        tuple(
            sorted(
                wire_of[(row + down, column + across)]
                for down, across in offsets
                if (row + down, column + across) in wire_of
            )
        )
        for row, column in faces
    )
    side = tuple(sorted(wire_of[vertex] for vertex in vertices if vertex[1] == 0))

    width = len(vertices)
    checks = tuple(
        tuple(1 if wire in support else 0 for wire in range(width))
        for support in supports
    )
    logical = (tuple(1 if wire in side else 0 for wire in range(width)),)
    return CssCode(
        hz=checks,
        hx=checks,
        lz=logical,
        lx=logical,
        distance_search_weight=int(distance),
    )


def toric_code(linear_size: int) -> CssCode:
    """Return the square-lattice toric code of the requested linear size.

    A toric code is written down on a torus rather than on a patch. The lattice is
    a ``linear_size``-by-``linear_size`` grid whose opposite sides are identified,
    every vertex carries one X-type stabilizer over the four edges meeting it, and
    every face carries one Z-type stabilizer over the four edges around it. The
    identification is what the family is: a periodic lattice has no boundary, so
    no face is cut down by one and every check has weight four at every size.

    **The patch is a torus, so the code carries two logical qubits rather than
    one.** A logical operator is a cycle of edges that wraps one of the two
    directions without bounding a face, and each direction carries one of each
    family. The Z-type family below is therefore two operators -- the ring of
    horizontal edges in row zero and the ring of vertical edges in column zero --
    and the X-type family is the matching pair of cuts, which cross the first ring
    and the second ring respectively. This is the first record in this package
    whose matrices leave ``k = 2``: ``css_code_matrices`` reports four logical
    operators for it, and a memory circuit built from it declares two observables.
    ``hz`` and ``hx`` are not one matrix here, because a star spans the edges at a
    vertex and a face spans the edges around a face; the two families are
    transposes of each other in the lattice sense and are computed from the same
    edge numbering rather than kept as two tables in step.

    **The linear size is the distance, and the distance is proved rather than
    stated.** The record handed back is a :class:`CssCode`, so ``distance``,
    ``x_distance`` and ``z_distance`` are searched for over the matrices derived
    here, and the search is given the linear size as its bound. That bound is the
    whole cost of this function, because the search tries every wire subset up to
    the bound: measured, the record builds in 0.00 s at size two, three and four,
    0.34 s at five and 23.1 s at six, on 2 * size ** 2 data qubits. It is
    therefore a way to write down the small tori an experiment is built from and
    not a way to write down a large one, and a caller who wants a larger lattice
    wants a decoder that reads its distance from the lattice instead -- which is
    the work of its own row rather than this route's.

    Args:
        linear_size: The number of vertices along one side of the torus, an
            integer of at least two. The patch has ``2 * linear_size ** 2`` data
            qubits, ``2 * linear_size ** 2`` checks and two logical qubits.

    Returns:
        The code record, with two logical qubits and both family distances equal
        to ``linear_size``.

    Raises:
        TypeError: If ``linear_size`` is not an integer.
        ValueError: If ``linear_size`` is below two, or if no logical operator of
            a family reaches the requested weight.

    Examples:
        >>> torus = toric_code(2)
        >>> torus.num_data_qubits, torus.num_ancilla_qubits, torus.distance
        (8, 8, 2)
        >>> torus.stabilizers[0] == Pauli(z_wires=(0, 1, 4, 6))
        True
    """

    if isinstance(linear_size, bool) or not isinstance(linear_size, Integral):
        raise TypeError("toric linear size must be an integer")
    if linear_size < 2:
        raise ValueError(
            f"the torus of linear size {linear_size} has no face and therefore no "
            "check, so it is not a code"
        )
    size = int(linear_size)
    width = 2 * size * size

    def edge(side: int, row: int, column: int) -> int:
        """The wire of one edge, with both torus coordinates wrapped."""

        return side * size * size + (row % size) + size * (column % size)

    def matrix_row(support: frozenset[int]) -> tuple[int, ...]:
        return tuple(1 if wire in support else 0 for wire in range(width))

    # A star is the four edges at a vertex: the two horizontal edges to its right
    # and the two vertical edges below it. A face is the four edges around a face:
    # the two horizontal edges of its top row and the two vertical edges of its
    # left column. Both are read off the one edge numbering above, so wrapping is
    # the modulus in that function and not a special case here.
    stars = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row, column),
                    edge(0, row, column + 1),
                    edge(1, row, column),
                    edge(1, row + 1, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    faces = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row - 1, column),
                    edge(0, row, column),
                    edge(1, row, column - 1),
                    edge(1, row, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    row_ring = frozenset(edge(0, 0, column) for column in range(size))
    column_ring = frozenset(edge(1, row, 0) for row in range(size))
    column_cut = frozenset(edge(0, row, 0) for row in range(size))
    row_cut = frozenset(edge(1, 0, column) for column in range(size))
    return CssCode(
        hz=faces,
        hx=stars,
        lz=(matrix_row(row_ring), matrix_row(column_ring)),
        lx=(matrix_row(column_cut), matrix_row(row_cut)),
        distance_search_weight=size,
    )


__all__ = (
    "CodeCheck",
    "CssCode",
    "RepetitionCode",
    "SteaneCode",
    "StabilizerCode",
    "toric_code",
    "triangular_colour_code",
)
