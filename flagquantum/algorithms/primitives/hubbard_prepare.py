"""Prepare the selection state of the square-lattice Hubbard model."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareHubbard.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/prepare_hubbard.py

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from math import isfinite, sqrt

from ...circuit import Circuit
from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import RotationCall
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.arithmetic.bounded_increment import (
    bounded_increment_ir,
    bounded_increment_requirements,
    source_bounded_increment_cost,
)
from ...compiler.registers.bits.bitwise import _validate_wires, xor_ir
from ...compiler.registers.control.multi_control import multi_and_ir
from ...compiler.registers.layout.hubbard_signatures import (
    hubbard_preparation_signature,
)
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import (
    opaque_register_type,
)
from ...core.ir.program.registers import Register
from ...dynamic import DynamicCircuit
from .prepare_oracle import PrepareOracle


def _hubbard_normalization(dimension: int, t: float, u: float) -> float:
    modes = 2 * dimension**2
    return 2 * modes * t + (modes * u) // 2


def _hubbard_preparation_angles(
    dimension: int, t: float, u: float
) -> tuple[float, float]:
    """Retain the frozen source's numerator order, floor and NumPy arccos."""
    import numpy as np

    modes = 2 * dimension**2
    return (
        float(2 * np.arccos(sqrt(t * modes / _hubbard_normalization(dimension, t, u)))),
        float(2 * np.arccos(np.sqrt(1 / 5))),
    )


def hubbard_selection_layout(dimension: int) -> RegisterLayout:
    """Return the common PREPARE and SELECT address order."""
    width = (dimension - 1).bit_length()
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


@dataclass(frozen=True)
class PrepareHubbard(PrepareOracle):
    """Bind the source selection ports and its two retained direction bits."""

    dimension: int
    t: float
    u: float

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Hubbard PREPARE requires positive square side length")
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
        return _hubbard_normalization(self.dimension, self.t, self.u)

    @property
    def l1_norm_of_coeffs(self) -> float:
        return self.alpha

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        return tuple(hubbard_preparation_signature(self.dimension))[:-1]

    @cached_property
    def junk_registers(self) -> tuple[Register, ...]:
        return (Register("temp", opaque_register_type(2)),)

    @property
    def selection_layout(self) -> RegisterLayout:
        return hubbard_selection_layout(self.dimension)

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
        if self.dimension == 1:
            raise ValueError(
                "Locked Hubbard PREPARE resources are undefined for side one"
            )
        width = (self.dimension - 1).bit_length()
        ports = 4 * width + 6
        uniform = source_uniform_preparation_cost(self.dimension)
        increment = source_bounded_increment_cost(width, 2)
        swap = ComponentCost("CSwap", 2 * width + 1, GateCost(cswap=width))
        direction_swap = ComponentCost(
            "CSwap", 4 * width + 3, GateCost(cswap=2 * width + 1)
        )
        angle, _ = _hubbard_preparation_angles(self.dimension, self.t, self.u)
        family = classify_resource_gate(RotationCall("ry", angle))
        rotation_cost = GateCost(
            t=int(family == "t"),
            clifford=int(family == "clifford"),
            rotation=int(family == "rotation"),
        )
        return ComponentCost(
            "PrepareHubbard",
            ports + 2,
            calls=(
                CostCall(ComponentCost("Ry", 1, rotation_cost)),
                CostCall(ComponentCost("Controlled", 2, GateCost(rotation=1))),
                CostCall(uniform, 2, bystander_qubits=ports - width),
                CostCall(ComponentCost("Hadamard", 1, GateCost(clifford=1)), 2),
                CostCall(ComponentCost("XGate", 1, GateCost(clifford=1)), 3),
                CostCall(ComponentCost("CNOT", 2, GateCost(clifford=1)), 2 * width + 3),
                CostCall(
                    ComponentCost("CHadamard", 2, GateCost(clifford=4, rotation=2))
                ),
                CostCall(swap, 2),
                CostCall(increment, bystander_qubits=ports - increment.qubits),
                CostCall(
                    ComponentCost("MultiAnd", 5, GateCost(and_bloq=2, clifford=4))
                ),
                CostCall(direction_swap),
                CostCall(
                    ComponentCost("Adjoint", 5, GateCost(clifford=6, measurement=2))
                ),
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
        angle, conditional_angle = _hubbard_preparation_angles(
            self.dimension, self.t, self.u
        )
        angles = (
            Instruction(
                "ry",
                (v,),
                params={"theta": sign * angle},
            ),
            Instruction("cry", (v, u), params={"theta": sign * conditional_angle}),
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
        axis_swap = controlled_swap_ir(axis, qx, qy) if width else ()
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
