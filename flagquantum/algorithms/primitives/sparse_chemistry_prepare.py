# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/sparse/prepare.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse/prepare.py

"""Forward sparse chemistry PREPARE with caller-owned output registers."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.less_equal import (
    less_equal_ir,
    less_equal_requirements,
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...core.ir import CircuitIR, Instruction
from .sparse_chemistry_data import SparseChemistryData


@dataclass(frozen=True)
class PrepareSparseChemistry:
    """Prepare the locked sparse chemistry state from zero-initialized ports."""

    data: SparseChemistryData

    @property
    def selection_layout(self) -> RegisterLayout:
        return (
            ("d", self.data.address_width),
            ("sigma", self.data.probability_bits),
            ("alpha", 1),
            ("beta", 1),
            ("rot_aa", 1),
            ("swap_pq", 1),
            ("swap_rs", 1),
            ("swap_pqrs", 1),
        )

    @property
    def junk_layout(self) -> RegisterLayout:
        return (("less_than", 1), *self.data.lookup_output_layout)

    def requirements(self) -> tuple[int, int]:
        """Return reusable clean scratch and classical slots."""
        _, uniform_work, uniform_slots = uniform_preparation_requirements(
            len(self.data.indices)
        )
        lookup = self.data.lookup_table().workspace_size(0, (self.data.address_width,))
        compare_work, compare_slots = less_equal_requirements(
            self.data.probability_bits, self.data.probability_bits
        )
        return max(uniform_work, lookup, compare_work), max(
            uniform_slots, lookup, compare_slots
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> "_BoundPrepareSparseChemistry":
        """Bind selection, retained outputs and scratch for circuit.append."""
        logical = unpartition_register(
            ports, (*self.selection_layout, *self.junk_layout)
        )
        _validate_wires((*logical, *workspace))
        required, slots = self.requirements()
        if (
            len(workspace) != required
            or len(classical_bits) != slots
            or len(set(classical_bits)) != slots
        ):
            raise ValueError(
                "Sparse PREPARE requires matching scratch and distinct slots"
            )
        data = self.data
        rows = len(data.indices)
        _, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
        address = ports["d"]
        uniform = uniform_preparation_ir(
            rows, address, workspace[:uniform_work], classical_bits[:uniform_slots]
        )
        lookup_work = data.lookup_table().workspace_size(0, (len(address),))
        lookup = data.lookup_ir(
            address,
            {name: ports[name] for name, _ in data.lookup_output_layout},
            workspace[:lookup_work],
            classical_bits[:lookup_work],
        )
        sigma = ports["sigma"]
        flag = ports["less_than"][0]
        compare_work, compare_slots = less_equal_requirements(len(sigma), len(sigma))
        compare = less_equal_ir(
            ports["keep"],
            sigma,
            flag,
            workspace[:compare_work],
            classical_bits[:compare_slots],
        )

        def swap(control: int, left: str, right: str) -> tuple[Instruction, ...]:
            return controlled_swap_ir(control, ports[left], ports[right])

        flip = Instruction("x", (flag,))
        instructions = (
            *uniform,
            *lookup,
            *(Instruction("h", (wire,)) for wire in sigma),
            *compare,
            Instruction("cz", (flag, *ports["alt_theta"])),
            flip,
            Instruction("cz", (flag, *ports["theta"])),
            flip,
            *(
                gate
                for orbital in ("p", "q", "r", "s")
                for gate in swap(flag, f"alt_{orbital}", orbital)
            ),
            *swap(flag, "flag_1b", "alt_flag_1b"),
            *(
                Instruction("h", ports[name])
                for name in ("swap_pq", "swap_rs", "swap_pqrs")
            ),
            *swap(ports["swap_pqrs"][0], "p", "r"),
            *swap(ports["swap_pqrs"][0], "q", "s"),
            *swap(ports["swap_pq"][0], "p", "q"),
            *swap(ports["swap_rs"][0], "r", "s"),
        )
        return _BoundPrepareSparseChemistry(
            instructions, max((*logical, *workspace)) + 1
        )

    def resource_model(self) -> ComponentCost:
        """Return the locked forward cost, including its one-bit swap declaration."""
        data = self.data
        lookup = data.lookup_cost()
        mu = data.probability_bits
        orbital_width = (data.spatial_orbitals - 1).bit_length()
        width = sum(bits for _, bits in (*self.selection_layout, *self.junk_layout))
        before_lookup = sum(bits for _, bits in self.selection_layout) + 1
        bystanders = before_lookup - data.address_width
        compare_width = 2 * mu + 1
        compare = source_equal_width_less_equal_cost(mu)
        # The source dictionary merges the flag CSwap into CSwap(1) at width one.
        swaps = 8 * orbital_width + int(orbital_width != 1)
        return ComponentCost(
            "PrepareSparse",
            width,
            GateCost(cswap=swaps, clifford=mu + 7),
            (
                CostCall(
                    source_uniform_preparation_cost(len(data.indices)),
                    bystander_qubits=bystanders,
                ),
                CostCall(lookup, bystander_qubits=bystanders),
                CostCall(compare, bystander_qubits=width - compare_width),
            ),
        )


@dataclass(frozen=True)
class _BoundPrepareSparseChemistry:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Sparse PREPARE binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
