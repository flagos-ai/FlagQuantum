"""Select a Pauli string from a quantum address."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....errors import ValidationError


@dataclass(frozen=True)
class SelectPauliLCU:
    """Apply an indexed Pauli string, with an optional single control.

    Strings follow the target order and accept a leading minus sign.
    Selection is most-significant-bit first. Unused addresses apply identity.
    """

    selection_bitsize: int
    target_bitsize: int
    select_unitaries: Sequence[str]
    control_val: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "select_unitaries", tuple(self.select_unitaries))

    def append_to(
        self,
        circuit: Circuit,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        control: Sequence[int] = (),
    ) -> Circuit:
        """Bind the declared registers and append one logical instruction."""
        if (
            len(selection) != self.selection_bitsize
            or len(target) != self.target_bitsize
            or len(control) != int(self.control_val is not None)
        ):
            raise ValidationError("SELECT bindings do not match the declared widths.")
        return circuit.gate(
            "select_pauli_lcu",
            (*control, *selection, *target),
            control_width=len(control),
            selection_width=self.selection_bitsize,
            target_width=self.target_bitsize,
            select_unitaries=self.select_unitaries,
            control_val=self.control_val,
        )
