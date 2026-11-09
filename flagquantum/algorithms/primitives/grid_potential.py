"""Coulomb phases using clean squared distances and fixed-point Newton steps."""

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, ldexp

from ...circuit import Circuit
from ...compiler.registers.fixed_point.inverse_sqrt import (
    _with_inverse_sqrt_estimate_ir,
    inverse_sqrt_requirements,
)
from ...compiler.registers.fixed_point.squared_distance import (
    PointWires,
    squared_distance_ir,
    squared_distance_requirements,
)
from ...compiler.registers.layout.partition import unpartition_register
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .grid_cubic_inverse_sqrt import (
    grid_cubic_inverse_sqrt_requirements,
    with_grid_cubic_inverse_sqrt_ir,
)
from .grid_kinetic import _coordinate_width
from .grid_qvr import grid_qvr_ir


def _inverse_requirements(
    distance: int, precision: int, iterations: int, cubic: bool
) -> tuple[int, int]:
    if cubic:
        if type(iterations) is not int or iterations != 1:
            raise ValueError("Cubic grid potential requires one Newton update")
        return grid_cubic_inverse_sqrt_requirements(distance, precision)
    return inverse_sqrt_requirements(
        distance, source_fraction_bits=0, precision=precision, iterations=iterations
    )


def grid_potential_requirements(
    grid_points: int, *, precision: int, iterations: int, cubic: bool = False
) -> tuple[int, int]:
    """Return reusable clean qubits and feedback slots for one pair."""
    distance, distance_work, distance_slots = squared_distance_requirements(
        _coordinate_width(grid_points)
    )
    inverse_work, inverse_slots = _inverse_requirements(
        distance, precision, iterations, cubic
    )
    return distance + max(distance_work, inverse_work), max(
        distance_slots, inverse_slots
    )


@dataclass(frozen=True)
class _BoundGridPotential:
    system: tuple[PointWires, ...]
    pairs: tuple[tuple[int, int], ...]
    workspace: tuple[int, ...]
    precision: int
    iterations: int
    cubic: bool
    radians_per_unit: float
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Grid potential binding exceeds circuit width")
        if not self.pairs:
            return circuit
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Grid potential erasure requires a DynamicCircuit")
        first_slot = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        distance, distance_work, distance_slots = squared_distance_requirements(
            len(self.system[0][0])
        )
        inverse_work, inverse_slots = _inverse_requirements(
            distance, self.precision, self.iterations, self.cubic
        )
        result, scratch = self.workspace[:distance], self.workspace[distance:]
        scale = ldexp(self.radians_per_unit, -self.precision)
        inverse_scratch = scratch[:inverse_work]
        inverse_bits = tuple(range(first_slot, first_slot + inverse_slots))

        def consume(estimate: tuple[int, ...]) -> tuple[Instruction, ...]:
            return grid_qvr_ir(estimate, radians_per_unit=scale)

        if self.cubic:
            phase = with_grid_cubic_inverse_sqrt_ir(
                result,
                inverse_scratch,
                inverse_bits,
                precision=self.precision,
                consume=consume,
            )
        else:
            phase = _with_inverse_sqrt_estimate_ir(
                result,
                inverse_scratch,
                inverse_bits,
                source_fraction_bits=0,
                precision=self.precision,
                iterations=self.iterations,
                tuned=False,
                consume=consume,
            )
        instructions: list[Instruction] = []
        for left, right in self.pairs:
            evaluate = squared_distance_ir(
                self.system[left],
                self.system[right],
                result,
                scratch[:distance_work],
                tuple(range(first_slot, first_slot + distance_slots)),
            )
            instructions.extend((*evaluate, *phase, *evaluate))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                tuple(instructions),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )


class GridPotentialEnergy:
    """Apply a fixed-point inverse-distance phase to each unordered electron pair."""

    @staticmethod
    def make_on(
        *,
        grid_points: int,
        system: tuple[PointWires, ...],
        workspace: tuple[int, ...],
        radians_per_unit: float,
        precision: int,
        iterations: int = 1,
        cubic: bool = False,
        adjoint: bool = False,
    ) -> _BoundGridPotential:
        """Bind three unsigned big-endian axis words for each electron.

        Differences retain their sign until their magnitude is taken. Zero
        distance yields zero phase. The default uses the shared C1 seed.
        Select cubic interpolation and a corrected Newton update with
        `cubic=True`. Precision is the number of fractional bits.
        Workspace and classical slots are reused between pairs.
        """
        width = _coordinate_width(grid_points)
        if not system or any(
            len(point) != 3 or any(len(axis) != width for axis in point)
            for point in system
        ):
            raise ValueError(
                "Grid potential requires three equal-width axes per electron"
            )
        if (
            type(adjoint) is not bool
            or type(cubic) is not bool
            or not isfinite(radians_per_unit)
        ):
            raise ValueError(
                "Grid potential requires a finite phase and a boolean adjoint"
            )
        pairs = tuple(combinations(range(len(system)), 2))
        work, _ = (
            grid_potential_requirements(
                grid_points, precision=precision, iterations=iterations, cubic=cubic
            )
            if pairs
            else (0, 0)
        )
        wires = unpartition_register(
            {
                "system": tuple(
                    wire for point in system for axis in point for wire in axis
                ),
                "work": workspace,
            },
            (("system", len(system) * 3 * width), ("work", work)),
        )
        return _BoundGridPotential(
            system,
            tuple(reversed(pairs)) if adjoint else pairs,
            workspace,
            precision,
            iterations,
            cubic,
            -radians_per_unit if adjoint else radians_per_unit,
            max(wires) + 1,
        )
