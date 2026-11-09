"""Select a Pauli string from a quantum address."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....compiler._component_trace.resource_model import ComponentCost
from ....compiler.registers.lookup.bounded_pauli_select import (
    bounded_pauli_select_ir,
    source_bounded_pauli_select_cost,
)
from ....core.ir import CircuitIR, Instruction
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

    def make_on(
        self,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        control: Sequence[int] = (),
    ) -> _BoundSelectPauliLCU:
        """Bind qubit indices for use with :meth:`Circuit.append`."""
        return _BoundSelectPauliLCU(
            self, tuple(selection), tuple(target), tuple(control)
        )

    def make_bounded_on(
        self,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        workspace: Sequence[int],
        classical_bits: Sequence[int],
        control: Sequence[int] = (),
    ) -> _BoundedSelectPauliLCU:
        """Bind the source bounded stream with explicit work and feedback slots.

        Padded addresses can apply a populated row. Use this path when the
        preparation restricts the address to the source selection domain.
        """
        instruction = self._instruction(selection, target, control)
        return _BoundedSelectPauliLCU(
            instruction, tuple(workspace), tuple(classical_bits)
        )

    def append_to(
        self,
        circuit: Circuit,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        control: Sequence[int] = (),
    ) -> Circuit:
        """Bind the declared registers and append one logical instruction."""
        instruction = self._instruction(selection, target, control)
        return circuit.gate(instruction.name, instruction.wires, **instruction.params)

    def _instruction(
        self, selection: Sequence[int], target: Sequence[int], control: Sequence[int]
    ) -> Instruction:
        if (
            len(selection) != self.selection_bitsize
            or len(target) != self.target_bitsize
            or len(control) != int(self.control_val is not None)
        ):
            raise ValidationError("SELECT bindings do not match the declared widths.")
        return Instruction(
            "select_pauli_lcu",
            (*control, *selection, *target),
            {
                "control_width": len(control),
                "selection_width": self.selection_bitsize,
                "target_width": self.target_bitsize,
                "select_unitaries": self.select_unitaries,
                "control_val": self.control_val,
            },
        )


@dataclass(frozen=True)
class _BoundSelectPauliLCU:
    operation: SelectPauliLCU
    selection: tuple[int, ...]
    target: tuple[int, ...]
    control: tuple[int, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        return self.operation.append_to(
            circuit, selection=self.selection, target=self.target, control=self.control
        )


@dataclass(frozen=True)
class _BoundedSelectPauliLCU:
    instruction: Instruction
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]

    def emit(self) -> tuple[Instruction, ...]:
        return bounded_pauli_select_ir(
            self.instruction, self.workspace, self.classical_bits
        )

    def resource_model(self) -> ComponentCost:
        return source_bounded_pauli_select_cost(self.instruction)

    def append_to(self, circuit: Circuit) -> Circuit:
        gates = self.emit()
        if any(
            wire >= circuit.n_qubits
            for wire in (*self.instruction.wires, *self.workspace)
        ):
            raise ValidationError("Bounded Pauli SELECT binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
