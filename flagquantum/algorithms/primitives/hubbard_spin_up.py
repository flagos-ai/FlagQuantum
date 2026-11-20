"""Potential-controlled phase selection on the spin-up Hubbard sector."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HubbardSpinUpZ.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.lookup.unary_iteration import (
    append_nested_bounded_unary_tree,
)
from ...compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .hubbard_qubitization_resource_model import source_hubbard_spin_up_formula


@dataclass(frozen=True)
class HubbardSpinUpZ:
    """Apply Z at y * dimension + x in the second half of target when V is one.

    Coordinate words are MSB first. The optional external control combines
    with V, and every temporary predicate and measurement slot is reusable.
    """

    dimension: int
    controlled: bool = False
    control_value: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError(
                "Hubbard spin-up phase requires positive square side length"
            )
        if type(self.controlled) is not bool:
            raise ValueError("Hubbard spin-up control flag must be boolean")
        if type(self.control_value) is not int or self.control_value not in (0, 1):
            raise ValueError("Hubbard spin-up control value must be zero or one")
        if not self.controlled and self.control_value == 0:
            raise ValueError("Hubbard spin-up control on zero needs a control wire")

    @property
    def layout(self) -> RegisterLayout:
        coordinate = (self.dimension - 1).bit_length()
        control = (("control", 1),) if self.controlled else ()
        return (
            *control,
            ("V", 1),
            ("x", coordinate),
            ("y", coordinate),
            ("target", 2 * self.dimension**2),
        )

    def requirements(self) -> tuple[int, int]:
        """Return the clean control conjunction and coordinate predicates."""
        work = 2 * (self.dimension - 1).bit_length() + int(self.controlled)
        return work, work

    def source_resource_model(self) -> ComponentCost:
        """Return the source declaration plus the native zero-control X sandwich."""
        formula = source_hubbard_spin_up_formula(
            self.dimension, controlled=self.controlled
        )
        return ComponentCost(
            "HubbardSpinUpZ",
            int(formula.logical_qubits),
            GateCost(
                and_bloq=int(formula.and_bloq),
                clifford=int(formula.clifford) + 2 * int(self.control_value == 0),
                measurement=int(formula.measurement),
            ),
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        if (len(work), len(slots)) != self.requirements():
            raise ValueError("Hubbard spin-up phase requires matching work and slots")
        (v,) = ports["V"]
        control = ports["control"][0] if self.controlled else None
        predicate = work[0] if self.controlled else v
        start = int(self.controlled)
        gates = (
            list(and_compute_ir(v, control, predicate)) if control is not None else []
        )

        def apply_z(index: int, enabled: int) -> None:
            gates.append(
                Instruction("cz", (enabled, ports["target"][self.dimension**2 + index]))
            )

        # This leaf traverses y then x, unlike the Majorana address.
        append_nested_bounded_unary_tree(
            gates,
            predicate,
            (ports["y"], ports["x"]),
            (self.dimension, self.dimension),
            work[start:],
            slots[start:],
            apply_z,
            zero_selection=lambda: apply_z(0, predicate),
        )
        if control is not None:
            gates.extend(and_uncompute_ir(v, control, predicate, slots[0]))
        if control is not None and self.control_value == 0:
            flip = Instruction("x", (control,))
            return (flip, *gates, flip)
        return tuple(gates)

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundHubbardSpinUpZ":
        """Bind the phase involution for circuit.append(op)."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if len(workspace) != self.requirements()[0]:
            raise ValueError("Hubbard spin-up phase requires matching clean work")
        return _BoundHubbardSpinUpZ(
            self, dict(ports), workspace, max((*logical, *workspace)) + 1
        )


@dataclass(frozen=True)
class _BoundHubbardSpinUpZ:
    operation: HubbardSpinUpZ
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "Hubbard spin-up phase needs a DynamicCircuit of sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        count = self.operation.requirements()[1]
        gates = self.operation._emit(
            self.ports, self.workspace, tuple(range(start, start + count))
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
