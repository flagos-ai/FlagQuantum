"""Measurement-assisted unary iteration for structured Pauli SELECT."""

from __future__ import annotations

from ....core.ir import Instruction
from ....errors import ValidationError
from ....runtime.dynamic.circuit import DynamicCircuit


def _and(circuit: DynamicCircuit, left: int, right: int, work: int) -> None:
    """Compute a conjunction into a clean qubit with four T gates."""
    circuit.h(work).t(work)
    circuit.cx(left, work).cx(right, work)
    circuit.cx(work, left).cx(work, right)
    circuit.tdg(left).tdg(right).t(work)
    circuit.cx(work, left).cx(work, right)
    circuit.h(work).s(work)


def lower_select_pauli_lcu_to_unary_iteration(
    instruction: Instruction,
) -> DynamicCircuit:
    """Lower a SELECT using clean workspace and classical feedback.

    The operand qubits follow instruction order. Remaining qubits are clean
    workspace and return to zero. This lowering measures workspace and
    therefore requires a dynamic executor.
    """
    if instruction.name != "select_pauli_lcu" or instruction.matrix is not None:
        raise ValidationError("Expected a structured select_pauli_lcu instruction.")

    params = instruction.params
    width = params["selection_width"]
    controlled = params["control_width"]
    words = params["select_unitaries"]
    operands = len(instruction.wires)
    targets = range(controlled + width, operands)
    circuit = DynamicCircuit(operands + max(width + controlled - 1, 0))

    def apply_word(address: int, control: int | None) -> None:
        word = words[address]
        if word.startswith("-"):
            if control is None:
                target = targets[0]
                circuit.x(target).z(target).x(target).z(target)
            else:
                circuit.z(control)
        for target, pauli in zip(targets, word.removeprefix("-"), strict=True):
            if pauli == "I":
                continue
            if control is None:
                circuit.gate(pauli.lower(), target)
            else:
                circuit.gate("c" + pauli.lower(), (control, target))

    def visit(control: int, depth: int, start: int, size: int) -> None:
        if start >= len(words):
            return
        if size == 1:
            apply_word(start, control)
            return
        right = controlled + depth
        work = operands + depth - (1 - controlled)
        half = size // 2
        # Cache the left predicate, then toggle it into the right predicate.
        circuit.x(right)
        _and(circuit, control, right, work)
        circuit.x(right)
        visit(work, depth + 1, start, half)
        circuit.cx(control, work)
        visit(work, depth + 1, start + half, half)
        circuit.h(work).measure(work, classical_bit=work)
        circuit.conditional("cz", (control, right), classical_bit=work)
        circuit.reset(work)

    if controlled:
        if params["control_val"] == 0:
            circuit.x(0)
        visit(0, 0, 0, 1 << width)
        if params["control_val"] == 0:
            circuit.x(0)
    elif width:
        half = 1 << (width - 1)
        circuit.x(0)
        visit(0, 1, 0, half)
        circuit.x(0)
        visit(0, 1, half, half)
    else:
        apply_word(0, None)
    return circuit
