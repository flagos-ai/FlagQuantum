# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/sparse/prepare.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse/prepare.py

"""Sparse chemistry preparation and explicit reversal strategies."""

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
from ...compiler.registers.qroam_resource_model import (
    source_qroam_measurement_erase_cost,
)
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
        return self._bind(ports, workspace, classical_bits, adjoint=False)

    def make_coherent_adjoint_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> "_BoundPrepareSparseChemistry":
        """Reverse the preparation with explicit coherent QROAM uncomputation.

        Supply the same logical wires and their matching forward bank state.
        The source's abstract measurement-erasure cost is available separately
        through resource_model(adjoint=True).
        """
        return self._bind(ports, workspace, classical_bits, adjoint=True)

    def make_measurement_adjoint_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        bank_bits: tuple[int, ...],
    ) -> "_BoundPrepareSparseChemistry":
        """Reverse preparation and consume matching lookup banks by measurement.

        Provide one outcome slot per bank wire in dataset, block cell and
        big-endian wire order, separate from all traversal scratch slots.
        """
        return self._bind(
            ports, workspace, classical_bits, adjoint=True, bank_bits=bank_bits
        )

    def _bind(
        self,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        *,
        adjoint: bool,
        bank_bits: tuple[int, ...] | None = None,
    ) -> "_BoundPrepareSparseChemistry":
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
            rows,
            address,
            workspace[:uniform_work],
            classical_bits[:uniform_slots],
            adjoint=adjoint,
        )
        lookup_work = data.lookup_table().workspace_size(0, (len(address),))
        lookup_outputs = {name: ports[name] for name, _ in data.lookup_output_layout}
        if bank_bits is None:
            lookup = data.lookup_ir(
                address,
                lookup_outputs,
                workspace[:lookup_work],
                classical_bits[:lookup_work],
                uncompute=adjoint,
            )
        else:
            if set(bank_bits).intersection(classical_bits):
                raise ValueError(
                    "Sparse PREPARE bank slots must be separate from scratch"
                )
            lookup = data.lookup_measurement_erase_ir(
                address,
                lookup_outputs,
                workspace[:lookup_work],
                classical_bits[:lookup_work],
                bank_bits,
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
        sample = tuple(Instruction("h", (wire,)) for wire in sigma)
        alternate_sign = (Instruction("cz", (flag, *ports["alt_theta"])),)
        original_sign = (flip, Instruction("cz", (flag, *ports["theta"])), flip)
        choose = (
            *(
                swap(flag, f"alt_{orbital}", orbital)
                for orbital in ("p", "q", "r", "s")
            ),
            swap(flag, "flag_1b", "alt_flag_1b"),
        )
        symmetry = (
            *(
                (Instruction("h", ports[name]),)
                for name in ("swap_pq", "swap_rs", "swap_pqrs")
            ),
            swap(ports["swap_pqrs"][0], "p", "r"),
            swap(ports["swap_pqrs"][0], "q", "s"),
            swap(ports["swap_pq"][0], "p", "q"),
            swap(ports["swap_rs"][0], "r", "s"),
        )
        stages = (
            uniform,
            lookup,
            sample,
            compare,
            alternate_sign,
            original_sign,
            *choose,
            *symmetry,
        )
        if adjoint:
            stages = (
                *reversed(symmetry),
                *reversed(choose),
                original_sign,
                alternate_sign,
                compare,
                sample[::-1],
                lookup,
                uniform,
            )
        instructions = tuple(gate for stage in stages for gate in stage)
        return _BoundPrepareSparseChemistry(
            instructions, max((*logical, *workspace)) + 1
        )

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        """Count the locked graph, using abstract measurement erasure for the adjoint."""
        data = self.data
        lookup = (
            source_qroam_measurement_erase_cost(
                data.lookup_table(), 0, tuple(bits for _, bits in data.target_layout)
            ).wrapper
            if adjoint
            else data.lookup_cost()
        )
        mu = data.probability_bits
        orbital_width = (data.spatial_orbitals - 1).bit_length()
        width = sum(bits for _, bits in (*self.selection_layout, *self.junk_layout))
        before_lookup = sum(bits for _, bits in self.selection_layout) + 1
        bystanders = before_lookup - data.address_width
        compare_width = 2 * mu + 1
        compare = source_equal_width_less_equal_cost(mu)
        # The source dictionary merges the flag CSwap into CSwap(1) at width one.
        swaps = 8 * orbital_width + int(orbital_width != 1)
        calls = (
            CostCall(
                source_uniform_preparation_cost(len(data.indices)),
                bystander_qubits=bystanders,
            ),
            CostCall(lookup, bystander_qubits=bystanders),
            CostCall(compare, bystander_qubits=width - compare_width),
        )
        return ComponentCost(
            "PrepareSparse†" if adjoint else "PrepareSparse",
            width,
            GateCost(cswap=swaps, clifford=mu + 7),
            calls[::-1] if adjoint else calls,
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
