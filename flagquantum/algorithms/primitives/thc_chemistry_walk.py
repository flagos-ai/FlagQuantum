# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source cost composition adapted from Qualtran v0.7.0 THC walk_operator.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/walk_operator.py

"""THC SELECT followed by a reflection about the complete prepared signal."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...runtime.dynamic.circuit import DynamicCircuit
from .multiplexers.select_thc import SelectTHC, _Network
from .multiplexers.thc_resource_model import select_thc_cost
from .thc_chemistry_prepare import PrepareTHCChemistry


@dataclass(frozen=True)
class THCChemistryWalk:
    """Bind a walk with caller-supplied SELECT rotation networks.

    Retain all PREPARE outputs as logical signal wires between steps.
    Use `select_workspace_qubits` to select the existing eight-bit
    lookup path. Zero uses exact rotations. Measurement-assisted
    SELECT uses slots above those reserved for the reflection.
    """

    prepare: PrepareTHCChemistry
    networks: tuple[_Network, ...]
    one_body_networks: tuple[_Network, ...]
    select_workspace_qubits: int = 0
    measurement_erasure: bool = False

    def __post_init__(self) -> None:
        data = self.prepare.data
        if (
            len(self.networks) != data.num_mu
            or len(self.one_body_networks) != data.num_spin_orb // 2
        ):
            raise ValueError(
                "THC walk rotation tables must match the prepared addresses"
            )
        if type(self.measurement_erasure) is not bool or (
            type(self.select_workspace_qubits) is not int
            or self.select_workspace_qubits < 0
        ):
            raise ValueError(
                "THC walk requires valid SELECT workspace and erasure settings"
            )
        if self.measurement_erasure and not self.select_workspace_qubits:
            raise ValueError("THC walk measurement erasure requires SELECT workspace")

    @property
    def alpha(self) -> float:
        return self.prepare.data.normalization

    @property
    def layout(self) -> RegisterLayout:
        orbitals = self.prepare.data.num_spin_orb // 2
        return (
            *self.prepare.selection_layout,
            *self.prepare.junk_layout,
            ("sys_a", orbitals),
            ("sys_b", orbitals),
        )

    def requirements(self) -> tuple[int, int]:
        """Return reusable clean quantum work and reflection classical slots."""
        prep_work, prep_slots = self.prepare.requirements()
        width = sum(size for _, size in self.layout[:-2])
        reflect_work, reflect_slots = prepare_reflection_requirements(
            width, controlled=False
        )
        return max(self.select_workspace_qubits, prep_work + 1, reflect_work), max(
            prep_slots, reflect_slots
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundTHCChemistryWalk":
        """Bind a full-signal walk step or its inverse for circuit.append."""
        if type(adjoint) is not bool:
            raise ValueError("THC walk direction must be boolean")
        logical = unpartition_register(ports, self.layout)
        work, slots = self.requirements()
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError("THC walk requires matching work wires and slots")
        _validate_wires((*logical, *workspace))
        _validate_wires(classical_bits)
        select = SelectTHC.make_on(
            mu=ports["mu"],
            nu=ports["nu"],
            succ=ports["succ"][0],
            nu_eq_mp1=ports["nu_eq_mp1"][0],
            plus_mn=ports["plus_mn"][0],
            plus_a=ports["plus_a"][0],
            plus_b=ports["plus_b"][0],
            sys_a=ports["sys_a"],
            sys_b=ports["sys_b"],
            networks=self.networks,
            one_body_networks=self.one_body_networks,
            workspace=workspace[: self.select_workspace_qubits],
            measurement_erasure=self.measurement_erasure,
        )
        prep_work, prep_slots = self.prepare.requirements()
        prep_ports = {name: ports[name] for name, _ in self.layout[:-2]}
        prep_workspace = workspace[1 : 1 + prep_work]
        prep_bits = classical_bits[:prep_slots]
        forward = self.prepare.make_on(
            prep_ports, workspace=prep_workspace, classical_bits=prep_bits
        ).instructions
        inverse = self.prepare.make_on(
            prep_ports, workspace=prep_workspace, classical_bits=prep_bits, adjoint=True
        ).instructions
        signal = logical[: -self.prepare.data.num_spin_orb]
        reflect_work, reflect_slots = prepare_reflection_requirements(
            len(signal), controlled=False
        )
        reflection = prepare_reflection_ir(
            signal,
            workspace[:reflect_work],
            classical_bits[:reflect_slots],
            prepare=forward,
            unprepare=inverse,
            global_phase=-1,
        )
        return _BoundTHCChemistryWalk(
            select,
            reflection,
            max((*logical, *workspace)) + 1,
            max(classical_bits) + 1,
            adjoint,
        )

    def source_resource_model(
        self, *, num_bits_theta: int = 8, kr1: int = 1, kr2: int = 1
    ) -> ComponentCost:
        """Count the locked walk with its narrower reflection and opaque erase."""
        data = self.prepare.data
        selection = sum(width for _, width in self.prepare.selection_layout)
        width = selection + data.num_spin_orb
        select = select_thc_cost(
            data.num_mu,
            data.num_spin_orb,
            num_bits_theta,
            data.probability_bits,
            kr1=kr1,
            kr2=kr2,
        )
        block = ComponentCost("SelectBlockEncoding", width, calls=(CostCall(select),))
        reflection = self._source_reflection(
            self.prepare.source_resource_model(),
            self.prepare.source_resource_model(adjoint=True),
        )
        return ComponentCost(
            "QubitizationWalkOperator",
            width,
            calls=(
                CostCall(block),
                CostCall(reflection, bystander_qubits=data.num_spin_orb),
            ),
        )

    def source_gate_cost(
        self, *, num_bits_theta: int = 8, kr1: int = 1, kr2: int = 1
    ) -> GateCost:
        """Count the locked walk gates when the source peak is unavailable."""
        data = self.prepare.data
        select = select_thc_cost(
            data.num_mu,
            data.num_spin_orb,
            num_bits_theta,
            data.probability_bits,
            kr1=kr1,
            kr2=kr2,
        )
        reflection = self._source_reflection(
            ComponentCost("PrepareTHC", 0, self.prepare.source_gate_cost()),
            ComponentCost(
                "PrepareTHC†", 0, self.prepare.source_gate_cost(adjoint=True)
            ),
        )
        return estimate_component_resources(select).gates.plus(
            estimate_component_resources(reflection).gates, 1
        )

    def _source_reflection(
        self, prepare: ComponentCost, unprepare: ComponentCost
    ) -> ComponentCost:
        return source_uncontrolled_prepare_reflection_cost(
            sum(width for _, width in self.prepare.selection_layout),
            sum(width for _, width in self.prepare.junk_layout),
            prepare,
            unprepare,
            mark_junk_width=self.prepare.data.address_width + 2,
        )


@dataclass(frozen=True)
class _BoundTHCChemistryWalk:
    select: SelectTHC
    reflection: tuple[Instruction, ...]
    required_qubits: int
    measurement_start: int
    adjoint: bool

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("THC walk binding exceeds circuit width")
        if self.select.measurement_erasure and not isinstance(circuit, DynamicCircuit):
            raise ValueError("THC walk measurement erasure requires a DynamicCircuit")
        fragment = DynamicCircuit(
            circuit.n_qubits, dtype=circuit.dtype, bsz=circuit.bsz
        )
        reflection = CircuitIR(
            circuit.n_qubits, self.reflection, metadata={"batch_size": circuit.bsz}
        ).validate()
        if self.adjoint:
            fragment.compose(reflection)
        first_bit = self.measurement_start
        if self.select.measurement_erasure:
            first_bit = max(
                first_bit,
                validate_classical_control(
                    circuit.to_ir().instructions, n_qubits=circuit.n_qubits
                ),
            )
        # On its prepared domain SELECT is an involution. Swap the two
        # stages for the walk adjoint, keeping the same SELECT operation.
        self.select._append_to(fragment, measurement_start=first_bit)
        if not self.adjoint:
            fragment.compose(reflection)
        return circuit.compose(fragment)
