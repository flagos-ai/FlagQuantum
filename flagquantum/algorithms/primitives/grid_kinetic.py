# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Resource graph adapted from the Qualtran v0.7.0 grid kinetic term.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/kinetic.py

"""Unsigned momentum-square phases on a three-dimensional grid."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.arithmetic.arithmetic_resource_model import (
    source_sum_of_squares_cost,
)
from ...compiler.registers.arithmetic.integer_square import (
    sum_of_squares_ir,
    sum_of_squares_requirements,
)
from ...compiler.registers.layout.partition import unpartition_register
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .grid_qvr import grid_qvr_ir, source_grid_qvr_cost


def _coordinate_width(grid_points: int) -> int:
    if type(grid_points) is not int or grid_points < 1:
        raise ValueError("Grid requires a positive number of points per axis")
    return (grid_points - 1).bit_length() + 1


def source_grid_kinetic_cost(electrons: int, grid_points: int) -> ComponentCost:
    """Model the ordered source square sum, QVR and Free for each electron."""
    if type(electrons) is not int or electrons < 1:
        raise ValueError("Grid kinetic energy requires a positive electron count")
    coordinate = _coordinate_width(grid_points)
    system = electrons * 3 * coordinate
    result = 2 * coordinate + 2
    electron = (
        CostCall(
            source_sum_of_squares_cost(coordinate, 3),
            bystander_qubits=system - 3 * coordinate,
        ),
        CostCall(source_grid_qvr_cost(result), bystander_qubits=system),
        CostCall(ComponentCost("Free", result), bystander_qubits=system),
    )
    return ComponentCost("KineticEnergy", system, calls=electron * electrons)


@dataclass(frozen=True)
class _BoundGridKinetic:
    coordinates: tuple[tuple[tuple[int, ...], ...], ...]
    result: tuple[int, ...]
    work: tuple[int, ...]
    phase: tuple[Instruction, ...]
    required_qubits: int
    cost: ComponentCost

    def resource_model(self) -> ComponentCost:
        return self.cost

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Grid kinetic binding exceeds circuit width")
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Grid kinetic erasure requires a DynamicCircuit")
        slot = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        instructions: list[Instruction] = []
        for electron in self.coordinates:
            square = sum_of_squares_ir(
                electron, self.result, self.work, measurement_bit=slot
            )
            instructions.extend((*square, *self.phase, *square))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                tuple(instructions),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )


class GridKineticEnergy:
    """Phase each electron by the sum of the squares of three unsigned words."""

    @staticmethod
    def make_on(
        *,
        grid_points: int,
        system: tuple[tuple[tuple[int, ...], ...], ...],
        result: tuple[int, ...],
        workspace: tuple[int, ...],
        radians_per_unit: float,
        adjoint: bool = False,
    ) -> _BoundGridKinetic:
        """Bind (electron, axis, big-endian wire) and reusable clean storage.

        The source sizes each coordinate with an extra bit and hands it to
        an unsigned square sum. All bit patterns follow that interpretation.
        The native square XOR is applied again after the phase to clear the
        result. The workspace is clean at each phase boundary.
        """
        cost = source_grid_kinetic_cost(len(system), grid_points)
        coordinate = _coordinate_width(grid_points)
        if any(
            len(electron) != 3 or any(len(axis) != coordinate for axis in electron)
            for electron in system
        ):
            raise ValueError(
                "Grid kinetic coordinates require three equal-width axis words"
            )
        if type(adjoint) is not bool:
            raise ValueError("Grid kinetic adjoint must be boolean")
        result_width, work_width = sum_of_squares_requirements(coordinate, 3)
        wires = unpartition_register(
            {
                "system": tuple(
                    wire for electron in system for axis in electron for wire in axis
                ),
                "result": result,
                "work": workspace,
            },
            (("system", cost.qubits), ("result", result_width), ("work", work_width)),
        )
        return _BoundGridKinetic(
            tuple(reversed(system)) if adjoint else system,
            result,
            workspace,
            grid_qvr_ir(
                result,
                radians_per_unit=-radians_per_unit if adjoint else radians_per_unit,
            ),
            max(wires) + 1,
            cost,
        )
