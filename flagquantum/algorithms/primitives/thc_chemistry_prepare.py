# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source preparation graph adapted from Qualtran v0.7.0 PrepareTHC.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/prepare.py

"""THC alias preparation with consistent one-body flags and a coherent inverse."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.contiguous_index import (
    contiguous_index_columns_ir,
    contiguous_index_columns_workspace_size,
    source_contiguous_index_cost,
)
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.equality import equality_constant_ir
from ...compiler.registers.less_equal import (
    less_equal_ir,
    less_equal_requirements,
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.multi_control import multi_control_x_ir
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.qroam_resource_model import (
    source_qroam_measurement_erase_cost,
)
from ...core.ir import CircuitIR, Instruction
from .thc_chemistry_data import THCChemistryData
from .thc_uniform_preparation import THCUniformPreparation


@dataclass(frozen=True)
class PrepareTHCChemistry:
    """Prepare the signed THC alias signal on zero-initialized logical ports.

    The contiguous address records the original row. The selected word
    and alternate word may exchange roles during alias sampling. The
    one-body flag describes the selected pair after sampling, while the
    retained QROAM banks, address and symmetry predicate are logical outputs.
    """

    data: THCChemistryData

    @property
    def selection_layout(self) -> RegisterLayout:
        width = self.data.num_mu.bit_length()
        return (
            ("mu", width),
            ("nu", width),
            ("plus_mn", 1),
            ("plus_a", 1),
            ("plus_b", 1),
            ("sigma", self.data.probability_bits),
            ("rot", 1),
            ("succ", 1),
            ("nu_eq_mp1", 1),
        )

    @property
    def junk_layout(self) -> RegisterLayout:
        return (
            ("s", self.data.address_width),
            ("less_than", 1),
            ("extra_ctrl", 1),
            *self.data.lookup_output_layout,
        )

    def requirements(self) -> tuple[int, int]:
        """Return clean quantum work and reusable measurement slots."""
        data = self.data
        width = data.num_mu.bit_length()
        uniform_work, uniform_slots = THCUniformPreparation(
            data.num_mu, data.num_spin_orb
        ).requirements()
        lookup = data.lookup_table().workspace_size(0, (data.address_width,))
        compare_work, compare_slots = less_equal_requirements(
            data.probability_bits, data.probability_bits
        )
        return max(
            uniform_work,
            lookup,
            compare_work,
            contiguous_index_columns_workspace_size(width),
        ), max(uniform_slots, lookup, compare_slots)

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundPrepareTHCChemistry":
        """Bind preparation or its coherent inverse for circuit.append."""
        if type(adjoint) is not bool:
            raise ValueError("THC preparation direction must be boolean")
        logical = unpartition_register(
            ports, (*self.selection_layout, *self.junk_layout)
        )
        work, slots = self.requirements()
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError("THC PREPARE requires matching work wires and slots")
        _validate_wires((*logical, *workspace))
        _validate_wires(classical_bits)

        data = self.data
        mu, nu, address = ports["mu"], ports["nu"], ports["s"]
        uniform = THCUniformPreparation(data.num_mu, data.num_spin_orb)
        uniform_work, uniform_slots = uniform.requirements()
        uniform_stage = uniform.make_on(
            {name: ports[name] for name, _ in uniform.layout},
            workspace=workspace[:uniform_work],
            classical_bits=classical_bits[:uniform_slots],
            adjoint=adjoint,
        ).instructions
        index_stage = contiguous_index_columns_ir(
            mu,
            nu,
            address,
            workspace[: contiguous_index_columns_workspace_size(len(mu))],
            measurement_bit=classical_bits[0],
        )
        lookup_work = data.lookup_table().workspace_size(0, (len(address),))
        lookup = data.lookup_ir(
            address,
            {name: ports[name] for name, _ in data.lookup_output_layout},
            workspace[:lookup_work],
            classical_bits[:lookup_work],
            uncompute=adjoint,
        )
        sigma = ports["sigma"]
        (alias_flag,) = ports["less_than"]
        compare_work, compare_slots = less_equal_requirements(len(sigma), len(sigma))
        compare = less_equal_ir(
            ports["keep"],
            sigma,
            alias_flag,
            workspace[:compare_work],
            classical_bits[:compare_slots],
        )
        sample = tuple(Instruction("h", (wire,)) for wire in sigma)
        flip = Instruction("x", (alias_flag,))
        signs = (
            Instruction("cz", (*ports["alt_theta"], alias_flag)),
            flip,
            Instruction("cz", (alias_flag, *ports["theta"])),
            flip,
        )
        (one_body,) = ports["nu_eq_mp1"]
        # Erase the old flag while nu still holds the input coordinate, then
        # write it from the selected coordinate after the alias exchanges.
        update_flag = equality_constant_ir(
            nu,
            data.num_mu,
            one_body,
            workspace[: len(nu) - 1],
            classical_bits[: len(nu) - 1],
        )
        (symmetry,) = ports["extra_ctrl"]
        plus = tuple(
            Instruction("h", ports[name]) for name in ("plus_a", "plus_b", "plus_mn")
        )
        symmetrize = multi_control_x_ir(
            (one_body, ports["plus_a"][0]),
            (0, 1),
            symmetry,
            workspace[:1],
            classical_bits[:1],
        )
        stages = (
            uniform_stage,
            index_stage,
            lookup,
            sample,
            compare,
            signs,
            update_flag,
            controlled_swap_ir(alias_flag, ports["alt_mu"], mu),
            controlled_swap_ir(alias_flag, ports["alt_nu"], nu),
            update_flag,
            compare,
            plus,
            symmetrize,
            controlled_swap_ir(symmetry, mu, nu),
        )
        ordered = reversed(stages) if adjoint else stages
        gates = tuple(gate for stage in ordered for gate in stage)
        return _BoundPrepareTHCChemistry(gates, max((*logical, *workspace)) + 1)

    def source_gate_cost(self, *, adjoint: bool = False) -> GateCost:
        """Count the locked graph even if its peak width is unavailable."""
        return estimate_component_resources(self._source_model(adjoint=adjoint)).gates

    def source_resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        """Count the locked graph and width on its supported blocked layout."""
        if self.data.log_block_size == 0:
            raise ValueError("The unblocked source PREPARE cannot bind junk_target0_")
        return self._source_model(adjoint=adjoint)

    def _source_model(self, *, adjoint: bool) -> ComponentCost:
        if type(adjoint) is not bool:
            raise ValueError("THC preparation direction must be boolean")
        data = self.data
        width = data.num_mu.bit_length()
        uniform = THCUniformPreparation(
            data.num_mu, data.num_spin_orb
        ).source_resource_model()
        index = source_contiguous_index_cost(width, data.address_width)
        lookup = (
            source_qroam_measurement_erase_cost(
                data.lookup_table(source_order=True),
                0,
                tuple(size for _, size in data.target_layout),
            ).wrapper
            if adjoint
            else data.lookup_cost(source_order=True)
        )
        compare = source_equal_width_less_equal_cost(data.probability_bits)
        swap = ComponentCost("CSwap", 1 + 2 * width, GateCost(cswap=width))
        symmetry = ComponentCost(
            "MultiControlX", 4, GateCost(and_bloq=1, clifford=2, measurement=1)
        )
        logical = sum(bits for _, bits in (*self.selection_layout, *self.junk_layout))
        banks = sum(bits for _, bits in data.lookup_output_layout)
        before_lookup = logical - banks
        calls = (
            CostCall(uniform, bystander_qubits=before_lookup - (2 * width + 3)),
            CostCall(index, bystander_qubits=before_lookup - index.qubits),
            CostCall(lookup, bystander_qubits=before_lookup - data.address_width),
            CostCall(compare, 2, logical - (2 * data.probability_bits + 1)),
            CostCall(swap, 3, logical - swap.qubits),
            CostCall(symmetry, bystander_qubits=logical - 3),
        )
        return ComponentCost(
            "PrepareTHC†" if adjoint else "PrepareTHC",
            logical,
            GateCost(clifford=data.probability_bits + 7),
            calls[::-1] if adjoint else calls,
        )


@dataclass(frozen=True)
class _BoundPrepareTHCChemistry:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("THC PREPARE binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
