"""Independent historical qec.surface rotated frame record."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from ...pauli import Pauli
from .._compat import _retain_module
from ..checks import CodeCheck, ancilla_band_counts


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

    Data qubits occupy indices ``0..distance**2 - 1``, indexed so that lattice
    site ``(i, j)`` is the qubit ``j * distance + i``. Ancillas follow the data
    qubits in lattice order. ``(i, j)`` is a data qubit for ``0 <= i, j <= distance - 1``
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
                stabilizer = Pauli(x_qubits=support)
                cnot_qubits = tuple((ancilla, wire) for wire in support)
            else:
                stabilizer = Pauli(z_qubits=support)
                cnot_qubits = tuple((wire, ancilla) for wire in support)
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
        return (Pauli(z_qubits=tuple(range(self.distance))),)


_retain_module("flagquantum.qec.surface", _surface_ancilla_sites, RotatedSurfaceCode)
