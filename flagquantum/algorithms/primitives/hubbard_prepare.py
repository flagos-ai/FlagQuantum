"""Prepare the selection state of the square-lattice Hubbard model."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareHubbard.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/prepare_hubbard.py

from collections.abc import Mapping
from dataclasses import dataclass
from math import acos, isfinite, sqrt

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bitwise import _validate_wires, xor_ir
from ...compiler.registers.bounded_increment import (
    bounded_increment_ir,
    bounded_increment_requirements,
    source_bounded_increment_cost,
)
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.multi_control import multi_and_ir
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit


@dataclass(frozen=True)
class PrepareHubbard:
    """Bind the source selection ports and its two retained direction bits."""

    dimension: int
    t: float
    u: float

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 2:
            raise ValueError("Hubbard PREPARE requires square side length at least two")
        if (
            not isfinite(self.t)
            or not isfinite(self.u)
            or min(self.t, self.u) < 0
            or not isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("Hubbard PREPARE requires finite nonnegative weights")

    @property
    def alpha(self) -> float:
        modes = 2 * self.dimension**2
        return 2 * modes * self.t + (modes * self.u) // 2

    @property
    def selection_layout(self) -> RegisterLayout:
        width = (self.dimension - 1).bit_length()
        return (
            ("U", 1),
            ("V", 1),
            ("p_x", width),
            ("p_y", width),
            ("alpha", 1),
            ("q_x", width),
            ("q_y", width),
            ("beta", 1),
        )

    @property
    def layout(self) -> RegisterLayout:
        return (*self.selection_layout, ("temp", 2))

    def requirements(self) -> tuple[int, int]:
        width, uniform_work, uniform_slots = uniform_preparation_requirements(
            self.dimension
        )
        shift_work, shift_slots = bounded_increment_requirements(
            width, self.dimension, 2
        )
        return max(uniform_work, shift_work, 2), max(uniform_slots, shift_slots, 2)

    def source_resource_model(self) -> ComponentCost:
        """Estimate the locked preparation, including the opaque ModAddK cost."""
        width = (self.dimension - 1).bit_length()
        ports = 4 * width + 6
        uniform = source_uniform_preparation_cost(self.dimension)
        increment = source_bounded_increment_cost(width, 2)
        local = GateCost(
            cswap=4 * width + 1,
            and_bloq=2,
            clifford=22 + 2 * width,
            rotation=4,
            measurement=2,
        )
        return ComponentCost(
            "PrepareHubbard",
            ports + 2,
            local,
            (
                CostCall(uniform, 2, bystander_qubits=ports - width),
                CostCall(increment, bystander_qubits=ports - increment.qubits),
            ),
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundPrepareHubbard":
        """Bind the preparation or its inverse for circuit.append."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if len(workspace) != self.requirements()[0] or type(adjoint) is not bool:
            raise ValueError("Hubbard PREPARE requires matching work and direction")
        return _BoundPrepareHubbard(
            self, dict(ports), workspace, adjoint, max((*logical, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        px, py, qx, qy = (ports[name] for name in ("p_x", "p_y", "q_x", "q_y"))
        (u,), (v,) = ports["U"], ports["V"]
        (alpha,), (beta,) = ports["alpha"], ports["beta"]
        axis, direction = ports["temp"]
        width, uniform_work, uniform_slots = uniform_preparation_requirements(
            self.dimension
        )
        shift_work, shift_slots = bounded_increment_requirements(
            width, self.dimension, 2
        )
        sign = -1 if adjoint else 1
        angles = (
            Instruction(
                "ry",
                (v,),
                params={
                    "theta": sign
                    * 2
                    * acos(sqrt(self.t * 2 * self.dimension**2 / self.alpha))
                },
            ),
            Instruction("cry", (v, u), params={"theta": sign * 2 * acos(sqrt(1 / 5))}),
        )
        phase_select = (
            Instruction("cx", (u, v)),
            Instruction("x", (beta,)),
            Instruction("x", (v,)),
            Instruction("ch", (v, alpha)),
            Instruction("cx", (v, beta)),
            Instruction("x", (v,)),
        )
        p, q = (*px, *py, alpha), (*qx, *qy, beta)
        axis_swap = controlled_swap_ir(axis, qx, qy)
        shift = bounded_increment_ir(
            qx,
            self.dimension,
            work[:shift_work],
            slots[:shift_slots],
            controls=(u, v),
            control_values=(0, 0),
            adjoint=adjoint,
        )
        controls, values = (u, v, direction), (0, 0, 1)
        direction_swap = (
            *multi_and_ir(controls, values, work[:2]),
            *controlled_swap_ir(work[1], p, q),
            *multi_and_ir(
                controls, values, work[:2], classical_bits=slots[:2], uncompute=True
            ),
        )
        stages = (
            angles[::-1] if adjoint else angles,
            uniform_preparation_ir(
                self.dimension,
                px,
                work[:uniform_work],
                slots[:uniform_slots],
                adjoint=adjoint,
            ),
            uniform_preparation_ir(
                self.dimension,
                py,
                work[:uniform_work],
                slots[:uniform_slots],
                adjoint=adjoint,
            ),
            (Instruction("h", (axis,)), Instruction("h", (direction,))),
            phase_select[::-1] if adjoint else phase_select,
            xor_ir(p, q),
            axis_swap,
            shift,
            axis_swap,
            direction_swap,
        )
        return tuple(
            gate
            for stage in (reversed(stages) if adjoint else stages)
            for gate in stage
        )


@dataclass(frozen=True)
class _BoundPrepareHubbard:
    prepare: PrepareHubbard
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "Hubbard PREPARE needs a DynamicCircuit of sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        slots = tuple(range(start, start + self.prepare.requirements()[1]))
        gates = self.prepare._emit(self.ports, self.workspace, slots, self.adjoint)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
