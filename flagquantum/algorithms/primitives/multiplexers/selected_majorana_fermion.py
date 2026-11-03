"""Register-indexed Jordan–Wigner operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
        """Append one logical instruction for deferred lowering."""
        qubits = (*self.control, *self.selection, *self.target)
        if any(type(q) is not int or not 0 <= q < circuit.n_qubits for q in qubits):
            raise ValidationError("SELECT qubits must be valid circuit indices.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("SELECT registers must use disjoint qubits.")
        if not 0 < len(self.target) <= 2 ** len(self.selection):
            raise ValidationError("The selection register must address every target.")
        if self.target_gate not in ("X", "Y", "Z"):
            raise ValidationError("target_gate must be X, Y, or Z.")
        return circuit.gate(
            "selected_majorana_fermion",
            qubits,
            params={
                "selection_width": len(self.selection),
                "target_width": len(self.target),
                "control_width": len(self.control),
                "target_gate": self.target_gate,
            },
        )
