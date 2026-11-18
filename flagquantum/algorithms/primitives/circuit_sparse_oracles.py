"""Native sparse oracle providers from explicit forward and adjoint snapshots."""

from dataclasses import dataclass
from functools import cached_property

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.primitives.circuit_binding import bind_primitive_circuit
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...core.value_math import SymbolicReal
from ._block_binding import _validate_block_work


def _storage(circuits: tuple[CircuitIR, CircuitIR], ports: int) -> tuple[int, int]:
    requirements: list[tuple[int, int]] = []
    for circuit in circuits:
        if not isinstance(circuit, CircuitIR):
            raise TypeError("Oracle snapshots must be owned CircuitIR values")
        if circuit.n_wires < ports:
            raise ValueError("Oracle snapshot is narrower than its logical ports")
        slots = validate_classical_control(
            circuit.instructions, n_qubits=circuit.n_wires
        )
        bind_primitive_circuit(
            circuit, tuple(range(circuit.n_wires)), classical_bits=tuple(range(slots))
        )
        requirements.append((circuit.n_wires - ports, slots))
    return max(work for work, _ in requirements), max(
        slots for _, slots in requirements
    )


def _validate(width: int, swaps: tuple[int, int]) -> None:
    if type(width) is not int or width < 1:
        raise ValueError("Oracle index width must be a positive integer")
    if len(swaps) != 2 or any(type(n) is not int or n < 0 for n in swaps):
        raise ValueError("Oracle binding swap counts need both nonnegative directions")


def _emit(
    circuit: CircuitIR,
    through: tuple[int, ...],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    required: tuple[int, int],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    if type(adjoint) is not bool or (len(work), len(slots)) != required:
        raise ValueError(
            "Oracle needs its declared work, result slots and Boolean direction"
        )
    _validate_block_work(through, work, slots)
    count = validate_classical_control(circuit.instructions, n_qubits=circuit.n_wires)
    return bind_primitive_circuit(
        circuit,
        (*through, *work[: circuit.n_wires - len(through)]),
        classical_bits=slots[:count],
    )


def _model(body: ComponentCost, circuit: CircuitIR, swaps: int) -> ComponentCost:
    if not swaps:
        return body
    binding = ComponentCost(
        "ReturnedWireBinding", circuit.n_wires, GateCost(clifford=swaps)
    )
    return ComponentCost(
        "ImportedOracle", circuit.n_wires, calls=(CostCall(body), CostCall(binding))
    )


@dataclass(frozen=True)
class CircuitColumnOracle:
    """Implement the existing column interface using owned snapshots and models.

    Snapshot wire order is ordinal, index, then clean work. Both directions
    are supplied explicitly; measurement instructions are never reversed.
    """

    system_bitsize: int
    num_nonzero: int
    forward: CircuitIR
    backward: CircuitIR
    models: tuple[ComponentCost, ComponentCost]
    binding_swaps: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        _validate(self.system_bitsize, self.binding_swaps)
        if (
            type(self.num_nonzero) is not int
            or not 1 <= self.num_nonzero <= 1 << self.system_bitsize
        ):
            raise ValueError("Column sparsity must fit the index width")
        if len(self.models) != 2 or any(
            not isinstance(model, ComponentCost) for model in self.models
        ):
            raise ValueError("Oracle needs separate forward and adjoint models")
        self.requirements()

    @cached_property
    def _required(self) -> tuple[int, int]:
        return _storage((self.forward, self.backward), 2 * self.system_bitsize)

    def requirements(self) -> tuple[int, int]:
        return self._required

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        if type(adjoint) is not bool:
            raise ValueError("Oracle model direction must be Boolean")
        return _model(
            self.models[int(adjoint)],
            self.backward if adjoint else self.forward,
            self.binding_swaps[int(adjoint)],
        )

    def emit(
        self,
        ordinal: tuple[int, ...],
        index: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        if len(ordinal) != self.system_bitsize or len(index) != self.system_bitsize:
            raise ValueError("Column operands must match the index width")
        return _emit(
            self.backward if adjoint else self.forward,
            (*ordinal, *index),
            work,
            slots,
            self.requirements(),
            adjoint,
        )


@dataclass(frozen=True)
class CircuitEntryOracle:
    """Bind a declared amplitude snapshot without transforming its coefficients.

    Wire order is flag, row, column, then clean work. Optional precision is
    metadata; the circuit itself determines the amplitude convention.
    """

    system_bitsize: int
    forward: CircuitIR
    backward: CircuitIR
    models: tuple[ComponentCost, ComponentCost]
    epsilon: SymbolicReal = 0
    binding_swaps: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        _validate(self.system_bitsize, self.binding_swaps)
        if not isinstance(self.epsilon, (int, float)):
            from sympy import Expr

            if not isinstance(self.epsilon, Expr):
                raise TypeError("Oracle precision must be a native scalar expression")
        if len(self.models) != 2 or any(
            not isinstance(model, ComponentCost) for model in self.models
        ):
            raise ValueError("Oracle needs separate forward and adjoint models")
        self.requirements()

    @cached_property
    def _required(self) -> tuple[int, int]:
        return _storage((self.forward, self.backward), 1 + 2 * self.system_bitsize)

    def requirements(self) -> tuple[int, int]:
        return self._required

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        if type(adjoint) is not bool:
            raise ValueError("Oracle model direction must be Boolean")
        return _model(
            self.models[int(adjoint)],
            self.backward if adjoint else self.forward,
            self.binding_swaps[int(adjoint)],
        )

    def emit(
        self,
        flag: int,
        row: tuple[int, ...],
        column: tuple[int, ...],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        if len(row) != self.system_bitsize or len(column) != self.system_bitsize:
            raise ValueError("Entry operands must match the index width")
        return _emit(
            self.backward if adjoint else self.forward,
            (flag, *row, *column),
            work,
            slots,
            self.requirements(),
            adjoint,
        )


@dataclass(frozen=True)
class CircuitSqrtEntryOracle(CircuitEntryOracle):
    """Mark a square-root amplitude declaration using the shared execution path."""
