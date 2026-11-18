"""Relocate an owned primitive fragment's quantum and classical operands."""

from ....core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ....core.classical_dataflow import ClassicalXor, parse_classical_xor
from ....core.ir import CircuitIR, Instruction
from .swap import swap_bit_ir


def _binding(values: tuple[int, ...], count: int, kind: str) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != count
        or any(type(value) is not int or value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            f"{kind} binding requires {count} distinct nonnegative integers"
        )


def bind_primitive_circuit(
    circuit: CircuitIR,
    quantum: tuple[int, ...],
    *,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Bind a self-contained instruction fragment to explicit caller storage.

    Maps cover every quantum wire and the zero-based prefix of classical slots.
    Quantum and classical spaces are independent. Input fragments may not read
    external classical bits or declare observables/static result nodes: those
    require program-level binding. No work allocation or slot allocation occurs.
    """
    circuit.validate()
    if circuit.observables or circuit.measurements:
        raise ValueError(
            "Primitive binding cannot drop program-level result declarations"
        )
    width = validate_classical_control(circuit.instructions, n_qubits=circuit.n_wires)
    _binding(quantum, circuit.n_wires, "Quantum")
    _binding(classical_bits, width, "Classical")
    result: list[Instruction] = []
    for gate in circuit.instructions:
        metadata = dict(gate.metadata)
        parameters = dict(gate.params)
        if gate.name == "classical_xor":
            assignment = parse_classical_xor(gate)
            bound = ClassicalXor(
                tuple(classical_bits[slot] for slot in assignment.sources),
                classical_bits[assignment.target],
                assignment.constant,
            ).instruction()
            parameters = dict(bound.params)
        if "classical_bit" in metadata:
            metadata["classical_bit"] = classical_bits[metadata["classical_bit"]]
        clauses = instruction_condition_clauses(gate)
        if clauses:
            bound_clauses = tuple(
                sorted(
                    (
                        tuple(
                            sorted(
                                (classical_bits[slot], expected)
                                for slot, expected in clause
                            )
                        )
                        for clause in clauses
                    ),
                    key=lambda clause: (len(clause), clause),
                )
            )
            metadata.pop("conditions", None)
            metadata.pop("condition_clauses", None)
            if len(bound_clauses) == 1:
                metadata["conditions"] = bound_clauses[0]
            else:
                metadata["condition_clauses"] = bound_clauses
        result.append(
            Instruction(
                gate.name,
                tuple(quantum[wire] for wire in gate.wires),
                parameters,
                matrix=gate.matrix,
                metadata=metadata,
            )
        )
    instructions = tuple(result)
    validate_classical_control(instructions, n_qubits=max(quantum) + 1)
    return instructions


def restore_output_order_ir(output_order: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Move the old value at output_order[i] to physical wire i.

    The complete permutation includes temporary wires. Its explicit swaps are
    physical binding overhead, separate from a source's logical wire renaming.
    """
    if any(type(value) is not int for value in output_order) or sorted(
        output_order
    ) != list(range(len(output_order))):
        raise ValueError("Output order must be a complete wire permutation")
    locations = list(range(len(output_order)))
    positions = locations.copy()
    gates: list[Instruction] = []
    for destination, source in enumerate(output_order):
        current = positions[source]
        if current == destination:
            continue
        displaced = locations[destination]
        gates.extend(swap_bit_ir(destination, current))
        locations[destination], locations[current] = source, displaced
        positions[source], positions[displaced] = destination, current
    return tuple(gates)
