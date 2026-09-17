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
    """One stabilizer check with its ancilla and its data-to-ancilla CNOTs.

    Each entry of ``cnot_wires`` is a ``(control, target)`` pair. The control is
    a data wire and the target is the check's ancilla.
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
        if any(control == self.ancilla_wire for control, _ in self.cnot_wires):
            raise ValueError("check CNOTs must control data wires, not the ancilla")
        if any(target != self.ancilla_wire for _, target in self.cnot_wires):
            raise ValueError("check CNOTs must target the declared ancilla")
        controls = tuple(sorted(control for control, _ in self.cnot_wires))
        if controls != self.stabilizer.support:
            raise ValueError("check CNOT controls must match the stabilizer support")


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


__all__ = ("CodeCheck", "RepetitionCode", "StabilizerCode")
