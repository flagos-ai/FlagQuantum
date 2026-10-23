"""Checkerboard-conjugated ZXXZ surface code."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from ...pauli import Pauli
from ..checks import CodeCheck, ancilla_band_counts
from .rotated import _surface_ancilla_sites


def _zxxz_checkerboard(distance: int) -> frozenset[int]:
    """Return the qubits a ZXXZ patch Hadamards, in the rotated patch's labelling.

    The patch is the rotated surface patch with a Hadamard on one sublattice of
    its data qubits: the checkerboard, whose site ``(i, j)`` is flipped when
    ``i + j`` is odd. Every check is therefore the rotated check with each of its
    qubits conjugated, which swaps that qubit's factors rather than the operator's
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
    """Return ``pauli`` with a Hadamard applied to every qubit on ``board``.

    A Hadamard maps ``X`` to ``Z`` on the qubit it acts on, so a qubit carrying a
    factor it is flipped on changes which list it appears in and a qubit it is not
    flipped on does not. Qubits off the board keep their factor and qubits on the
    board exchange theirs, which is the whole of the conjugation.
    """

    x_qubits = frozenset(pauli.x_qubits)
    z_qubits = frozenset(pauli.z_qubits)
    return Pauli(
        x_qubits=tuple(sorted((x_qubits - board) | (z_qubits & board))),
        z_qubits=tuple(sorted((z_qubits - board) | (x_qubits & board))),
    )


@dataclass(frozen=True)
class ZxxzSurfaceCode:
    """The rotated surface patch read out in the ZXXZ frame, of odd ``distance``.

    This is the rotated surface patch of the same distance with a Hadamard on the
    data checkerboard, so it occupies the same qubits, declares the same number of
    ancillas in the same lattice order, and has the same stabilizer group up to a
    local Clifford conjugation. What changes is which factor each check carries on
    each of its qubits: conjugating a rotated check's qubits by the Hadamard swaps
    the factors on the qubits that sit on the board, so a check whose support had
    one qubit on the board has that qubit's factor exchanged and a check whose
    support was entirely on or entirely off the board keeps its type. On the
    rotated patch's own assignment **every check has at least one qubit on the
    board and at least one off it**, so every check comes out mixed -- an X-and-Z
    product on one ancilla -- and none is pure.

    **Why the parameters are the rotated patch's.** A Hadamard on a data qubit is a
    Clifford conjugation, so it maps the stabilizer group to an isomorphic group
    on the same qubits, preserves every operator's weight, and preserves which
    operators commute. The code therefore has the same ``n``, the same ``k``, the
    same check weights, and the same distance as the rotated patch of the same
    distance, and this record's ``distance`` is that number rather than a
    separately computed one. What the conjugation does change is the *type* of
    every operator, which is why the matrix route cannot state this family and the
    memory-circuit route can.

    **The declared logical pair is the two diagonals.** ``Z`` on the anti-diagonal
    and ``X`` on the main diagonal both commute with every check, neither lies in
    the stabilizer span, and they overlap at the patch centre -- exactly one qubit,
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
    def num_ancilla_x_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[0]

    @property
    def num_ancilla_z_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[1]

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
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
            rotated = Pauli(x_qubits=support) if is_x else Pauli(z_qubits=support)
            stabilizer = _conjugate_by_checkerboard(rotated, board)
            # ``CodeCheck`` reads the coupling as its stabilizer orders the
            # factors: the Z-factor pairs first, each reading a data qubit into the
            # ancilla, then the X-factor pairs, each reading the ancilla out into
            # a data qubit. A mixed check therefore uses both directions and one
            # ancilla, which is the shape the record describes.
            cnot_qubits = tuple(
                (wire, ancilla) for wire in stabilizer.z_qubits
            ) + tuple((ancilla, wire) for wire in stabilizer.x_qubits)
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=stabilizer,
                    ancilla_qubit=ancilla,
                    cnot_qubits=cnot_qubits,
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
        return (Pauli(z_qubits=downward), Pauli(x_qubits=upward))
