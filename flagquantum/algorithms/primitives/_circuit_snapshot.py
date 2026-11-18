"""Validate, bind and price explicitly directional native circuit snapshots."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.primitives.circuit_binding import bind_primitive_circuit
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ._block_binding import _validate_block_work


def _storage(circuits: tuple[CircuitIR, ...], ports: int) -> tuple[int, int]:
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
