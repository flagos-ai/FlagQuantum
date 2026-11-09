# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source resource graph adapted from Qualtran v0.7.0 UniformSuperpositionTHC.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/prepare.py

"""Uniform THC address preparation and the locked source resource graph."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import acos, pi, sin, sqrt

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.comparison.equality import equality_constant_ir
from ...compiler.registers.comparison.less_equal import (
    less_equal_ir,
    less_equal_requirements,
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.comparison.less_than import (
    less_than_constant_ir,
    less_than_constant_workspace_size,
    source_less_than_constant_cost,
)
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.ir import CircuitIR, Instruction


@dataclass(frozen=True)
class THCUniformPreparation:
    """Prepare equal positive amplitudes on the THC alias table's address domain.

    Two-body addresses satisfy mu <= nu < num_mu. One-body addresses have
    nu == num_mu and mu below the number of spatial orbitals. The source's
    port name nu_eq_mp1 denotes the one-body flag. On zero input, succ is
    one, rot is zero and all temporary wires are cleared.
    """

    num_mu: int
    num_spin_orb: int

    def __post_init__(self) -> None:
        if (
            type(self.num_mu) is not int
            or self.num_mu < 1
            or type(self.num_spin_orb) is not int
            or self.num_spin_orb < 2
            or self.num_spin_orb % 2
            or self.num_spin_orb // 2 > 1 << self.num_mu.bit_length()
        ):
            raise ValueError("THC dimensions must fit the unsigned address registers")

    @property
    def layout(self) -> RegisterLayout:
        width = self.num_mu.bit_length()
        return (("mu", width), ("nu", width), ("nu_eq_mp1", 1), ("succ", 1), ("rot", 1))

    def requirements(self) -> tuple[int, int]:
        """Return clean work and reusable measurement slots."""
        width = self.num_mu.bit_length()
        work, slots = less_equal_requirements(width, width)
        zero_work, zero_slots = prepare_reflection_requirements(
            2 * width + 1, controlled=False
        )
        return max(5 + max(work, width + 1), zero_work), max(slots, width, zero_slots)

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundTHCUniformPreparation":
        """Bind an exact amplification sequence or its inverse for circuit.append."""
        if type(adjoint) is not bool:
            raise ValueError("THC preparation direction must be boolean")
        logical = unpartition_register(ports, self.layout)
        work, slots = self.requirements()
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError(
                "THC uniform preparation requires matching work wires and slots"
            )
        _validate_wires((*logical, *workspace))
        _validate_wires(classical_bits)

        mu, nu = ports["mu"], ports["nu"]
        (rot,) = ports["rot"]
        (succ,) = ports["succ"]
        (one_body,) = ports["nu_eq_mp1"]
        below_mu, ordered, at_sentinel, below_spatial, good = workspace[:5]
        compare_work = workspace[5:]
        width = len(mu)
        spatial = self.num_spin_orb // 2

        def less_than(
            word: tuple[int, ...], threshold: int, target: int
        ) -> tuple[Instruction, ...]:
            needed = less_than_constant_workspace_size(width, threshold)
            return less_than_constant_ir(
                word,
                threshold,
                target,
                compare_work[:needed],
                classical_bits[:width] if needed else (),
            )

        lt = less_than(nu, self.num_mu, below_mu)
        le_work, le_slots = less_equal_requirements(width, width)
        le = less_equal_ir(
            mu, nu, ordered, compare_work[:le_work], classical_bits[:le_slots]
        )
        eq_args = (compare_work[: width - 1], classical_bits[: width - 1])
        sentinel = equality_constant_ir(nu, self.num_mu, at_sentinel, *eq_args)
        lt_spatial = less_than(mu, spatial, below_spatial)
        flags = (*lt, *le, *sentinel, *lt_spatial)
        unflags = (*lt_spatial, *sentinel, *le, *lt)
        triangle = Instruction("ccx", (below_mu, ordered, good))
        single = Instruction("ccx", (at_sentinel, below_spatial, good))
        mark = (
            *flags,
            triangle,
            single,
            Instruction("x", (good,)),
            *prepare_reflection_ir(
                (rot, good), compare_work[:2], classical_bits[:1], global_phase=1
            ),
            Instruction("x", (good,)),
            single,
            triangle,
            *unflags,
        )
        record = (
            *flags,
            triangle,
            single,
            Instruction("cx", (good, succ)),
            single,
            triangle,
            *unflags,
            *equality_constant_ir(nu, self.num_mu, one_body, *eq_args),
        )

        domain = self.num_mu * (self.num_mu + 1) // 2 + spatial
        capacity = 1 << (2 * width)
        # One exact Grover step needs initial success probability 1/4.
        # The triangular part alone occupies at least 1/8 of the address
        # cube, so two steps always suffice when one is impossible.
        rounds = 1 if 4 * domain >= capacity else 2
        overlap = sin(pi / (4 * rounds + 2))
        angle = 2 * acos(min(1.0, overlap * sqrt(capacity / domain)))
        hadamards = tuple(Instruction("h", (wire,)) for wire in (*mu, *nu))
        initialize = (*hadamards, Instruction("ry", (rot,), params={"theta": angle}))
        uninitialize = (Instruction("ry", (rot,), params={"theta": -angle}), *hadamards)
        zero_work, zero_slots = prepare_reflection_requirements(
            2 * width + 1, controlled=False
        )
        zero = prepare_reflection_ir(
            (*mu, *nu, rot),
            workspace[:zero_work],
            classical_bits[:zero_slots],
            global_phase=-1,
        )
        if adjoint:
            before, iteration, after = (
                record,
                (*uninitialize, *zero, *initialize, *mark),
                uninitialize,
            )
        else:
            before, iteration, after = (
                initialize,
                (*mark, *uninitialize, *zero, *initialize),
                record,
            )
        gates = (*before, *(iteration * rounds), *after)
        return _BoundTHCUniformPreparation(gates, max((*logical, *workspace)) + 1)

    def source_resource_model(self) -> ComponentCost:
        """Price the locked source graph, including its out-of-range sentinel check."""
        width = self.num_mu.bit_length()
        sentinel = self.num_mu + 1
        if sentinel >= 1 << width:
            raise ValueError(
                "The locked THC one-body sentinel does not fit its address width"
            )
        live = 2 * width + 7  # Signature and the four allocated flags.
        lt = source_less_than_constant_cost(width, self.num_mu)
        le = source_equal_width_less_equal_cost(width)
        # GreaterThanConstant has only a cost declaration. It points to
        # LessThanConstant with the same threshold in the pinned source.
        gt = source_less_than_constant_cost(width, self.num_spin_orb // 2)
        equality = ComponentCost(
            "EqualsAConstant",
            2 * width,
            GateCost(
                and_bloq=width - 1,
                clifford=width
                + (4 * (width - sentinel.bit_count()) if width > 2 else 0),
                measurement=width - 1,
            ),
        )
        success = ComponentCost(
            "MultiControlX", 6, GateCost(and_bloq=2, clifford=3, measurement=2)
        )
        first_mark, second_mark = (
            source_uncontrolled_prepare_reflection_cost(
                size,
                0,
                ComponentCost("PrepareIdentity", size),
                ComponentCost("PrepareIdentity", size),
            )
            for size in (4, 2 * width + 1)
        )
        return ComponentCost(
            "UniformSuperpositionTHC",
            live,
            GateCost(clifford=6 * width + 1, rotation=2, toffoli=4),
            (
                CostCall(lt, 4, live - width - 1),
                CostCall(le, 4, live - 2 * width - 1),
                CostCall(equality, 3, live - width - 1),
                CostCall(gt, 4, live - width - 1),
                CostCall(first_mark, bystander_qubits=live - 4),
                CostCall(second_mark, bystander_qubits=live - 2 * width - 1),
                CostCall(success, bystander_qubits=live - 4),
            ),
        )


@dataclass(frozen=True)
class _BoundTHCUniformPreparation:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("THC uniform preparation exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
