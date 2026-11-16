"""Register-indexed Jordan–Wigner operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....compiler._component_trace.resource_model import ComponentCost
from ....compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    source_bounded_majorana_cost,
)
from ....core.ir import CircuitIR, Instruction
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

    @classmethod
    def make_bounded_on(
        cls,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        workspace: Sequence[int],
        classical_bits: Sequence[int],
        control: Sequence[int] = (),
        target_gate: str = "Y",
    ) -> _BoundedMajoranaSelection:
        """Bind the source traversal with explicit clean work and feedback slots.

        The bounded address is big-endian and uses the source padding behavior.
        The work word starts with the parity carrier, followed by predicates.
        This entry supports zero or one positive control.
        """
        op = cls.make_on(
            selection=selection, target=target, control=control, target_gate=target_gate
        )
        if len(op.control) > 1:
            raise ValidationError("Bounded Majorana accepts at most one control.")
        return _BoundedMajoranaSelection(op, tuple(workspace), tuple(classical_bits))

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


@dataclass(frozen=True)
class _BoundedMajoranaSelection:
    operation: SelectedMajoranaFermion
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]

    def emit(self) -> tuple[Instruction, ...]:
        op = self.operation
        return bounded_majorana_ir(
            op.selection,
            op.target,
            op.control[0] if op.control else None,
            op.target_gate,
            self.workspace,
            self.classical_bits,
            selection_width=len(op.selection),
        )

    def resource_model(self) -> ComponentCost:
        op = self.operation
        return source_bounded_majorana_cost(
            len(op.target),
            op.target_gate,
            controlled=bool(op.control),
            selection_width=len(op.selection),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        op = self.operation
        if any(
            type(wire) is not int or not 0 <= wire < circuit.n_qubits
            for wire in (*op.control, *op.selection, *op.target, *self.workspace)
        ):
            raise ValidationError("Bounded Majorana bindings exceed circuit width.")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, self.emit(), metadata={"batch_size": circuit.bsz}
            ).validate()
        )
