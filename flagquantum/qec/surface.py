"""The rotated surface lattice, and the two frames a patch is read in.

A surface patch is the one family in this package that is *derived* rather than
declared: its checks are read off a lattice rule, so the rotated patch and the
ZXXZ patch are not two records that happen to resemble each other but one lattice
written down twice. The rotated patch is the anchor and the ZXXZ patch is the
rotated patch with a Hadamard on one sublattice of its data qubits, which is why
this module holds both and why the second one's ``distance`` is the first one's
number rather than a separately searched quantity.

The seam is the lattice. :mod:`flagquantum.qec.codes` holds the records a caller
supplies and the record a code arrives as, together with the catalogue that names
every family this package declares. This module holds the single ancilla
assignment both surface frames are cut from and the coupling that turns one of
them into a mixed-check patch, so the shared machinery has one home rather than
being reached across a catalogue that is about what a code *is*.

**A check's coupling is a consequence of its factors, not a choice.** The
conjugation that produces the ZXXZ frame swaps which factor a wire carries, and
because a factor decides which way a CNOT points, a mixed check ends up driving
one ancilla in both directions -- reading its Z factor in and its X factor out.
That is a property this module's records inherit from the lattice and the
conjugation; :class:`~flagquantum.qec.codes.CodeCheck` is where the rule is
enforced, and :mod:`flagquantum.qec.circuit` is the only place that has to emit
it.

This module describes checks, wires and observables. It does not choose a decoding
strategy, a noise model, or a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .codes import CodeCheck
from .pauli import Pauli


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


def _zxxz_checkerboard(distance: int) -> frozenset[int]:
    """Return the wires a ZXXZ patch Hadamards, in the rotated patch's labelling.

    The patch is the rotated surface patch with a Hadamard on one sublattice of
    its data qubits: the checkerboard, whose site ``(i, j)`` is flipped when
    ``i + j`` is odd. Every check is therefore the rotated check with each of its
    wires conjugated, which swaps that wire's factors rather than the operator's
    identity, and every check of the patch comes out mixed.

    The parity is load-bearing and not a convention. Hadamarding the other
    sublattice also turns every check mixed and also leaves ``k = 1``, but it
    leaves no pure Z-type representative of the conjugated logical observable, so
    a Z-basis memory experiment of that patch would have no observable to declare.
    """

    return frozenset(
        j * distance + i
        for i in range(distance)
        for j in range(distance)
        if (i + j) % 2 == 1
    )


def _conjugate_by_checkerboard(pauli: Pauli, board: frozenset[int]) -> Pauli:
    """Return ``pauli`` with a Hadamard applied to every wire on ``board``.

    A Hadamard maps ``X`` to ``Z`` on the wire it acts on, so a wire carrying a
    factor it is flipped on changes which list it appears in and a wire it is not
    flipped on does not. Wires off the board keep their factor and wires on the
    board exchange theirs, which is the whole of the conjugation.
    """

    x_wires = frozenset(pauli.x_wires)
    z_wires = frozenset(pauli.z_wires)
    return Pauli(
        x_wires=tuple(sorted((x_wires - board) | (z_wires & board))),
        z_wires=tuple(sorted((z_wires - board) | (x_wires & board))),
    )


@dataclass(frozen=True)
class ZxxzSurfaceCode:
    """The rotated surface patch read out in the ZXXZ frame, of odd ``distance``.

    This is the rotated surface patch of the same distance with a Hadamard on the
    data checkerboard, so it occupies the same wires, declares the same number of
    ancillas in the same lattice order, and has the same stabilizer group up to a
    local Clifford conjugation. What changes is which factor each check carries on
    each of its wires: conjugating a rotated check's wires by the Hadamard swaps
    the factors on the wires that sit on the board, so a check whose support had
    one wire on the board has that wire's factor exchanged and a check whose
    support was entirely on or entirely off the board keeps its type. On the
    rotated patch's own assignment **every check has at least one wire on the
    board and at least one off it**, so every check comes out mixed -- an X-and-Z
    product on one ancilla -- and none is pure.

    **Why the parameters are the rotated patch's.** A Hadamard on a data wire is a
    Clifford conjugation, so it maps the stabilizer group to an isomorphic group
    on the same wires, preserves every operator's weight, and preserves which
    operators commute. The code therefore has the same ``n``, the same ``k``, the
    same check weights, and the same distance as the rotated patch of the same
    distance, and this record's ``distance`` is that number rather than a
    separately computed one. What the conjugation does change is the *type* of
    every operator, which is why the matrix route cannot state this family and the
    memory-circuit route can.

    **The declared logical pair is the two diagonals.** ``Z`` on the anti-diagonal
    and ``X`` on the main diagonal both commute with every check, neither lies in
    the stabilizer span, and they overlap at the patch centre -- exactly one wire,
    which is what makes them a logical pair. The overlap is odd only when the
    distance is odd, which is why even distances are refused: the record declares
    *that* pair, and at an even distance the two diagonals can miss each other.

    A check of this record is measured by one ancilla in both directions, so the
    memory-circuit route needs no gadget beyond the one :class:`CodeCheck`
    describes; the per-check rate order the matrix route defines has no meaning
    here, so a per-check measurement-flip vector is refused and the uniform rate
    applies. This record describes the checks and the observables. It does not
    choose a decoding strategy, a noise model, or a threshold.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("ZXXZ-surface-code distance must be an integer")
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError(
                "ZXXZ-surface-code distance must be odd and at least three, "
                "because the declared logical pair is the two diagonals and they "
                "overlap at a single wire only on an odd patch"
            )

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
        board = _zxxz_checkerboard(distance)
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
            rotated = Pauli(x_wires=support) if is_x else Pauli(z_wires=support)
            stabilizer = _conjugate_by_checkerboard(rotated, board)
            # ``CodeCheck`` reads the coupling as its stabilizer orders the
            # factors: the Z-factor pairs first, each reading a data wire into the
            # ancilla, then the X-factor pairs, each reading the ancilla out into
            # a data wire. A mixed check therefore uses both directions and one
            # ancilla, which is the shape the record describes.
            cnot_wires = tuple((wire, ancilla) for wire in stabilizer.z_wires) + tuple(
                (ancilla, wire) for wire in stabilizer.x_wires
            )
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
        distance = self.distance
        upward = tuple(range(0, distance * distance, distance + 1))
        downward = tuple(
            row * (distance - 1) + (distance - 1) for row in range(distance)
        )
        return (Pauli(z_wires=downward), Pauli(x_wires=upward))


__all__ = (
    "RotatedSurfaceCode",
    "ZxxzSurfaceCode",
)
