"""Qubitization of square-lattice Hubbard selection with a complete signal."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Hubbard qubitization walk composition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/walk_operator.py

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR
from ...dynamic import DynamicCircuit
from .hubbard_prepare import PrepareHubbard
from .hubbard_select import SelectHubbard


@dataclass(frozen=True)
class HubbardWalk:
    """Apply SELECT then reflect about PREPARE's output state.

    The two direction bits are persistent signal ports. Prepare the complete
    signal before the first full-signal step. Set `reflect_directions=False`
    to reproduce the source mark on selection alone. In that mode the
    direction ports can start and end in zero, as in the locked walk.
    """

    prepare: PrepareHubbard
    reflect_directions: bool = True

    def __post_init__(self) -> None:
        if type(self.reflect_directions) is not bool:
            raise ValueError("Hubbard walk reflection mode must be boolean")

    @property
    def layout(self) -> RegisterLayout:
        return (*self.prepare.layout, ("target", 2 * self.prepare.dimension**2))

    @property
    def alpha(self) -> float:
        return self.prepare.alpha

    @property
    def reflection_layout(self) -> RegisterLayout:
        """Return the registers tested for zero after unpreparation."""
        return (
            self.prepare.layout
            if self.reflect_directions
            else self.prepare.selection_layout
        )

    def requirements(self) -> tuple[int, int]:
        """Return clean work and feedback slots reused across walk stages."""
        select_work, select_slots = SelectHubbard(self.prepare.dimension).requirements()
        prep_work, prep_slots = self.prepare.requirements()
        reflect_work, reflect_slots = prepare_reflection_requirements(
            sum(width for _, width in self.reflection_layout), controlled=False
        )
        return max(select_work, prep_work + 1, reflect_work), max(
            select_slots, prep_slots, reflect_slots
        )

    def source_resource_model(self) -> ComponentCost:
        """Estimate the locked walk with the source's narrower mark and junk."""
        width = sum(bits for _, bits in self.prepare.selection_layout)
        system = self.layout[-1][1]
        select = SelectHubbard(self.prepare.dimension).source_resource_model()
        block = ComponentCost(
            "SelectBlockEncoding", width + system, calls=(CostCall(select),)
        )
        prep = self.prepare.source_resource_model()
        reflection = source_uncontrolled_prepare_reflection_cost(width, 2, prep, prep)
        return ComponentCost(
            "QubitizationWalkOperator",
            width + system,
            calls=(CostCall(block), CostCall(reflection, bystander_qubits=system)),
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundHubbardWalk":
        """Bind the walk or its adjoint for circuit.append."""
        wires = unpartition_register(ports, self.layout)
        _validate_wires((*wires, *workspace))
        if len(workspace) != self.requirements()[0] or type(adjoint) is not bool:
            raise ValueError("Hubbard walk requires matching work and direction")
        return _BoundHubbardWalk(
            self, dict(ports), workspace, adjoint, max((*wires, *workspace)) + 1
        )


@dataclass(frozen=True)
class _BoundHubbardWalk:
    walk: HubbardWalk
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError("Hubbard walk needs a DynamicCircuit of sufficient width")
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        slots = tuple(range(start, start + self.walk.requirements()[1]))
        prepare = self.walk.prepare
        select = SelectHubbard(prepare.dimension)
        select_work, select_slots = select.requirements()
        # SELECT is a Hermitian signed Pauli permutation on the populated addresses.
        selection = select._emit(
            self.ports, self.workspace[:select_work], slots[:select_slots]
        )
        prep_work, prep_slots = prepare.requirements()
        prep_args = (self.ports, self.workspace[1 : 1 + prep_work], slots[:prep_slots])
        signal = tuple(
            wire for name, _ in self.walk.reflection_layout for wire in self.ports[name]
        )
        reflect_work, reflect_slots = prepare_reflection_requirements(
            len(signal), controlled=False
        )
        reflection = prepare_reflection_ir(
            signal,
            self.workspace[:reflect_work],
            slots[:reflect_slots],
            prepare=prepare._emit(*prep_args, False),
            unprepare=prepare._emit(*prep_args, True),
            global_phase=-1,
        )
        gates = (*reflection, *selection) if self.adjoint else (*selection, *reflection)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
