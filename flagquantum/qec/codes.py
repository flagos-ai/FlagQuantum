"""Code-independent stabilizer-code descriptions.

A code declares its wire layout, its checks, and its logical observables. The
frozen repetition profile keeps its own records; this module describes a code as
a value so that circuit generation, detector layout, and decoding can be derived
from it rather than pinned to one instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

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


__all__ = (
    "CodeCheck",
    "RepetitionCode",
    "RotatedSurfaceCode",
    "SteaneCode",
    "StabilizerCode",
)
