# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source model adapted from Qualtran v0.7.0 sparse chemistry walk_operator.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse/walk_operator.py

"""Sparse chemistry walk with an explicit complete preparation signal."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from .multiplexers.select_sparse import SelectSparse
from .qubitization_walk import _BoundQubitizationWalk
from .sparse_chemistry_prepare import PrepareSparseChemistry


@dataclass(frozen=True)
class SparseChemistryWalk:
    """Apply SELECT and reflect about the complete prepared signal.

    Keep all PREPARE outputs as logical ports between steps. SELECT uses the
    orbital and body-flag outputs, and the reflection includes those outputs
    and the remaining preparation registers. Supply populated orbital
    addresses to the bounded SELECT path. The reference resource model uses
    the source's narrower reflection and its separate auxiliary allocation.
    """

    prepare: PrepareSparseChemistry

    @property
    def num_spin_orb(self) -> int:
        return 2 * self.prepare.data.spatial_orbitals

    @property
    def alpha(self) -> float:
        return self.prepare.data.normalization

    @property
    def layout(self) -> RegisterLayout:
        return (
            *self.prepare.selection_layout,
            *self.prepare.junk_layout,
            ("sys", self.num_spin_orb),
        )

    def requirements(self) -> tuple[int, int]:
        """Return reusable clean quantum workspace and classical feedback slots."""
        select_work, select_slots = SelectSparse.source_requirements(self.num_spin_orb)
        prep_work, prep_slots = self.prepare.requirements()
        signal_width = sum(bits for _, bits in self.layout[:-1])
        reflection_work, reflection_slots = prepare_reflection_requirements(
            signal_width, controlled=False
        )
        return max(select_work, prep_work + 1, reflection_work), max(
            select_slots, prep_slots, reflection_slots
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
    ) -> _BoundQubitizationWalk:
        """Bind one complete-signal walk step and its optional adjoint."""
        if type(adjoint) is not bool:
            raise ValueError("Sparse chemistry walk direction must be boolean")
        logical = unpartition_register(ports, self.layout)
        work, slots = self.requirements()
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError(
                "Sparse chemistry walk requires matching work wires and slots"
            )
        _validate_wires((*logical, *workspace))
        _validate_wires(classical_bits)

        select_work, select_slots = SelectSparse.source_requirements(self.num_spin_orb)
        select = SelectSparse.make_source_on(
            self.num_spin_orb,
            {
                name: ports[name]
                for name, _ in SelectSparse.source_layout(self.num_spin_orb)
            },
            workspace=workspace[:select_work],
            classical_bits=classical_bits[:select_slots],
        ).emit(adjoint=adjoint)
        prep_ports = {name: ports[name] for name, _ in self.layout[:-1]}
        prep_work, prep_slots = self.prepare.requirements()
        prep_workspace = workspace[1 : 1 + prep_work]
        prep_bits = classical_bits[:prep_slots]
        forward = self.prepare.make_on(
            prep_ports, workspace=prep_workspace, classical_bits=prep_bits
        ).instructions
        inverse = self.prepare.make_coherent_adjoint_on(
            prep_ports, workspace=prep_workspace, classical_bits=prep_bits
        ).instructions
        signal = logical[: -self.num_spin_orb]
        reflection_work, reflection_slots = prepare_reflection_requirements(
            len(signal), controlled=False
        )
        reflection = prepare_reflection_ir(
            signal,
            workspace[:reflection_work],
            classical_bits[:reflection_slots],
            prepare=forward,
            unprepare=inverse,
            global_phase=-1,
        )
        gates = (*reflection, *select) if adjoint else (*select, *reflection)
        return _BoundQubitizationWalk(gates, max((*logical, *workspace)) + 1)

    def source_resource_model(self) -> ComponentCost:
        """Price the locked walk, including its abstract PREPARE adjoint."""
        select_layout = SelectSparse.source_layout(self.num_spin_orb)
        prep_layout = self.prepare.selection_layout
        selection_width = sum(bits for _, bits in prep_layout)
        width = sum((dict(select_layout) | dict(prep_layout)).values())
        select = SelectSparse.source_resource_model(self.num_spin_orb)
        block = ComponentCost(
            "SelectBlockEncoding", select.qubits, calls=(CostCall(select),)
        )
        reflection = source_uncontrolled_prepare_reflection_cost(
            selection_width,
            sum(bits for _, bits in self.prepare.junk_layout),
            self.prepare.resource_model(),
            self.prepare.resource_model(adjoint=True),
            mark_junk_width=1,  # The comparison flag passes through; QROAM banks do not.
        )
        return ComponentCost(
            "QubitizationWalkOperator",
            width,
            calls=(
                CostCall(block, bystander_qubits=width - block.qubits),
                CostCall(reflection, bystander_qubits=width - selection_width),
            ),
        )
