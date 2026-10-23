"""Steane code definitions."""

from __future__ import annotations

from dataclasses import dataclass

from ..pauli import Pauli
from ._compat import _retain_module
from .checks import CodeCheck, ancilla_band_counts

_STEANE_CHECK_SUPPORTS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)

_STEANE_LOGICAL_SUPPORT: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class SteaneCode:
    """The seven-qubit Steane code, declared by its checks and its logicals.

    Data qubits occupy qubits ``0..6`` and the six check ancillas follow them on
    qubits ``7..12``: checks ``0..2`` are Z-type and checks ``3..5`` are X-type,
    each with the same support, which is what makes the code Calderbank-Shor-
    Steane. The supports are the three non-zero parity constraints of the
    ``[7, 4, 3]`` Hamming code, so each check has weight four and the code has
    distance three.

    Both logical operators are declared, on the same three data qubits: ``Z`` on
    ``(0, 1, 2)`` and ``X`` on ``(0, 1, 2)``. They anticommute, since they overlap
    so an odd number of qubits flips the is a product of checks. This is the
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
        return tuple(range(7))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return tuple(range(7, 13))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(_STEANE_CHECK_SUPPORTS):
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((wire, ancilla) for wire in support),
                )
            )
        for offset, support in enumerate(_STEANE_CHECK_SUPPORTS):
            index = len(_STEANE_CHECK_SUPPORTS) + offset
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(x_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((ancilla, wire) for wire in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (
            Pauli(z_qubits=_STEANE_LOGICAL_SUPPORT),
            Pauli(x_qubits=_STEANE_LOGICAL_SUPPORT),
        )


_retain_module("flagquantum.qec.codes", SteaneCode)
