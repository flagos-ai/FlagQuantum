"""Register-indexed Jordan–Wigner operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ....circuit import Circuit
from ....errors import ValidationError


@dataclass(frozen=True)
class SelectedMajoranaFermion:
    """Bound coherent selection of a Jordan–Wigner Pauli string.

    The selection is big-endian. Address j applies Z on each preceding target
    and the specified Pauli on target j. Unused addresses act as identity.
    All supplied control qubits must be one to enable the operation.
    """

    selection: tuple[int, ...]
    target: tuple[int, ...]
    control: tuple[int, ...] = ()
    target_gate: str = "Y"

    @classmethod
    def make_on(
        cls,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        control: Sequence[int] = (),
        target_gate: str = "Y",
    ) -> SelectedMajoranaFermion:
        """Bind the operation using FlagQuantum qubit indices."""
        return cls(tuple(selection), tuple(target), tuple(control), target_gate)

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append one structured instruction using the dense reference executor."""
        qubits = (*self.control, *self.selection, *self.target)
        if any(type(q) is not int or not 0 <= q < circuit.n_qubits for q in qubits):
            raise ValidationError("SELECT qubits must be valid circuit indices.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("SELECT registers must use disjoint qubits.")
        if not 0 < len(self.target) <= 2 ** len(self.selection):
            raise ValidationError("The selection register must address every target.")
        if self.target_gate not in ("X", "Y", "Z"):
            raise ValidationError("target_gate must be X, Y, or Z.")
        if len(qubits) > 10:
            raise ValidationError(
                "The dense SELECT reference supports at most 10 qubits."
            )

        target_width = len(self.target)
        dimension = 1 << len(qubits)
        matrix = torch.eye(dimension, dtype=torch.complex128)
        for address in range(len(self.target)):
            mask = 1 << (target_width - 1 - address)
            start = dimension - (1 << (target_width + len(self.selection)))
            start += address << target_width
            for state in range(1 << target_width):
                parity = (state >> (target_width - address)).bit_count()
                bit = bool(state & mask)
                phase = (-1) ** parity
                column = start + state
                if self.target_gate == "Z":
                    matrix[column, column] = phase * (-1 if bit else 1)
                else:
                    matrix[column, column] = 0
                    matrix[column ^ mask, column] = (
                        phase
                        if self.target_gate == "X"
                        else phase * (-1j if bit else 1j)
                    )
        return circuit.gate(
            "selected_majorana_fermion",
            qubits,
            matrix=matrix,
            params={
                "selection_width": len(self.selection),
                "target_width": len(self.target),
                "control_width": len(self.control),
                "target_gate": self.target_gate,
            },
        )
