"""Code-independent stabilizer-code descriptions.

A code declares its wire layout, its checks, and its logical observables. The
frozen repetition profile keeps its own records; this module describes a code as
a value so that circuit generation, detector layout, and decoding can be derived
from it rather than pinned to one instance.

Two routes reach such a record. A family declares its own layout from a distance
or an index -- the repetition profile, the rotated surface patch and the Steane
code all do -- and :class:`CssCode` instead takes the four matrices a
Calderbank-Shor-Steane code is already written down as, which is the route a code
this package does not declare arrives by. Both satisfy the same protocol, so no
consumer can tell which one it was handed.
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

    Each entry of ``cnot_wires`` is a ``(control, target)`` pair, and which wire
    is which is fixed by the check type rather than left to the caller. A
    Z-type check couples every data wire in the stabilizer's support into an
    ancilla prepared in ``|0>``, so the data wire controls and the ancilla is
    the target. An X-type check couples the ancilla out into the same support
    with the ancilla prepared in ``|+>``, so the ancilla controls and the data
    wire is the target. Both gadgets leave the ancilla's Z-basis readout equal
    to the check's eigenvalue, which is what makes the two symmetric here.
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
        if self.stabilizer.x_wires and self.stabilizer.z_wires:
            raise ValueError(
                "check stabilizer must be pure X-type or pure Z-type, not a "
                "mixture: a mixture needs a second ancilla and a second CNOT "
                "direction, which this record does not describe"
            )
        if self.ancilla_wire < 0:
            raise ValueError("check ancilla wire must be non-negative")
        if not self.cnot_wires:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_wires:
            if control < 0 or target < 0:
                raise ValueError("check CNOT wires must be non-negative")
        if self.stabilizer.x_wires:
            self._validate_x_type()
        else:
            self._validate_z_type()

    def _validate_z_type(self) -> None:
        if any(control == self.ancilla_wire for control, _ in self.cnot_wires):
            raise ValueError("check CNOTs must control data wires, not the ancilla")
        if any(target != self.ancilla_wire for _, target in self.cnot_wires):
            raise ValueError("check CNOTs must target the declared ancilla")
        controls = tuple(sorted(control for control, _ in self.cnot_wires))
        if controls != self.stabilizer.support:
            raise ValueError("check CNOT controls must match the stabilizer support")

    def _validate_x_type(self) -> None:
        if any(target == self.ancilla_wire for _, target in self.cnot_wires):
            raise ValueError("check CNOTs must target data wires, not the ancilla")
        if any(control != self.ancilla_wire for control, _ in self.cnot_wires):
            raise ValueError("check CNOTs must be controlled by the declared ancilla")
        targets = tuple(sorted(target for _, target in self.cnot_wires))
        if targets != self.stabilizer.support:
            raise ValueError("check CNOT targets must match the stabilizer support")


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


def _surface_ancilla_sites(distance: int) -> tuple[tuple[int, int, bool], ...]:
    """Return the rotated-surface-code ancilla sites as ``(a, b, is_x)``.

    Data qubits sit on the lattice ``(i, j)`` with ``0 <= i, j <= distance - 1``
    and an ancilla sits on ``(a, b)`` with ``0 <= a, b <= distance``. An X-type
    ancilla needs an interior column and an odd lattice parity, a Z-type ancilla
    needs an interior row and an even lattice parity, which is the stabilizer
    assignment the rotated layout requires: the parity that would place an
    ancilla outside the patch is what truncates the boundary checks.

    Sites are reported in lattice order so the check and ancilla indices are
    reproducible.
    """

    sites: list[tuple[int, int, bool]] = []
    for a in range(distance + 1):
        for b in range(distance + 1):
            if 1 <= a <= distance - 1 and (a + b) % 2 == 1:
                sites.append((a, b, True))
            elif 1 <= b <= distance - 1 and (a + b) % 2 == 0:
                sites.append((a, b, False))
    return tuple(sites)


@dataclass(frozen=True)
class RotatedSurfaceCode:
    """The rotated surface code of odd ``distance`` read out as Z memory.

    Data qubits occupy wires ``0..distance**2 - 1``, indexed so that lattice
    site ``(i, j)`` is wire ``j * distance + i``. Ancillas follow the data wires
    in lattice order. ``(i, j)`` is a data qubit for ``0 <= i, j <= distance - 1``
    and ``(a, b)`` is an ancilla for ``0 <= a, b <= distance``; an X-type ancilla
    sits on an interior column with odd lattice parity, a Z-type ancilla on an
    interior row with even lattice parity, and its support is the up to four data
    qubits diagonally adjacent to it. That assignment is the rotated layout: a
    check that would fall outside the patch is truncated, which is what leaves
    the boundary checks at weight two.

    The declared logical observable is ``Z`` on the data row ``j == 0``, which is
    one complete row of the patch, and the two logical operators of the patch
    therefore read out as Z memory. The X-type checks are still measured every
    round, because they detect the Z errors this readout is vulnerable to; they
    are simply not deterministic in the initial all-zero state.

    This record describes the checks and the observable. It does not choose a
    decoding strategy, a noise model, or an error-correction threshold.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("rotated-surface-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("rotated-surface-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance * self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return len(_surface_ancilla_sites(self.distance))

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        first = self.num_data_qubits
        return tuple(first + index for index in range(self.num_ancilla_qubits))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        distance = self.distance
        checks: list[CodeCheck] = []
        for index, (a, b, is_x) in enumerate(_surface_ancilla_sites(distance)):
            ancilla = self.num_data_qubits + index
            support = tuple(
                sorted(
                    j * distance + i
                    for i in range(distance)
                    for j in range(distance)
                    if i in (a - 1, a) and j in (b - 1, b)
                )
            )
            if is_x:
                stabilizer = Pauli(x_wires=support)
                cnot_wires = tuple((ancilla, wire) for wire in support)
            else:
                stabilizer = Pauli(z_wires=support)
                cnot_wires = tuple((wire, ancilla) for wire in support)
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=stabilizer,
                    ancilla_wire=ancilla,
                    cnot_wires=cnot_wires,
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=tuple(range(self.distance))),)


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


__all__ = (
    "CodeCheck",
    "CssCode",
    "RepetitionCode",
    "RotatedSurfaceCode",
    "SteaneCode",
    "StabilizerCode",
)
