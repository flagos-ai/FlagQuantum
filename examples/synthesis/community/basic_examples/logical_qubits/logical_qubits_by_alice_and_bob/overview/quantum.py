"""Original seven-bit SWAP test with exact real-amplitude rotations."""

import math

import numpy as np
import torch

import flagquantum as fq

from .synthesis import approximate


def amplitudes() -> tuple[float, ...]:
    values = 1 - 2 * np.random.RandomState(12).rand(8)
    values /= np.linalg.norm(values)
    return tuple(float(value) for value in values)


def controlled_y(circuit: fq.Circuit, control: int, target: int, angle: float) -> None:
    circuit.ry(target, theta=angle / 2)
    circuit.cx(control, target)
    circuit.ry(target, theta=-angle / 2)
    circuit.cx(control, target)


def prepare(
    circuit: fq.Circuit, wires: tuple[int, int, int], values: tuple[float, ...]
) -> None:
    if (
        len(values) != 8
        or not all(math.isfinite(value) for value in values)
        or abs(sum(value * value for value in values) - 1) > 1e-12
    ):
        raise ValueError("The source input requires eight normalized real amplitudes")
    if len(set(wires)) != 3 or any(not 0 <= wire < circuit.n_qubits for wire in wires):
        raise ValueError("Preparation wires must be distinct and within the circuit")
    for depth in range(3):
        stride = 1 << (3 - depth)
        for prefix in range(1 << depth):
            branch = values[prefix * stride : (prefix + 1) * stride]
            half = stride // 2
            if depth == 2:
                left, right = branch
            else:
                left = math.sqrt(sum(value * value for value in branch[:half]))
                right = math.sqrt(sum(value * value for value in branch[half:]))
            angle = 2 * math.atan2(right, left)
            negative = tuple(
                wires[index]
                for index in range(depth)
                if not ((prefix >> (depth - index - 1)) & 1)
            )
            for wire in negative:
                circuit.x(wire)
            if depth == 0:
                circuit.ry(wires[0], theta=angle)
            elif depth == 1:
                controlled_y(circuit, wires[0], wires[1], angle)
            else:
                first, second, target = wires
                controlled_y(circuit, second, target, angle / 2)
                circuit.cx(first, second)
                controlled_y(circuit, second, target, -angle / 2)
                circuit.cx(first, second)
                controlled_y(circuit, first, target, angle / 2)
            for wire in reversed(negative):
                circuit.x(wire)


def swap_test() -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    values = amplitudes()
    prepare(circuit, (1, 2, 3), values)
    prepare(circuit, (4, 5, 6), values)
    circuit.h(0)
    for first, second in zip((1, 2, 3), (4, 5, 6), strict=True):
        circuit.cx(second, first)
        circuit.ccx(0, first, second)
        circuit.cx(second, first)
    circuit.h(0)
    return circuit


def synthesize(
    circuit: fq.Circuit, iterations: int
) -> tuple[fq.Circuit, tuple[float, ...]]:
    """Approximate each uncontrolled RY/RX; preserve every native entangling gate."""
    result = fq.Circuit(int(circuit.n_qubits), dtype=torch.complex128)
    errors = []
    for instruction in circuit.to_ir().instructions:
        if instruction.name in ("rx", "ry"):
            angle = float(instruction.params["theta"])
            axis = 1 if instruction.name == "rx" else 2
            quaternion = np.zeros(4, dtype=np.float64)
            quaternion[0], quaternion[axis] = math.cos(angle / 2), math.sin(angle / 2)
            fitted = approximate(quaternion, iterations)
            errors.append(fitted.error)
            for name in fitted.word:
                result.gate(name, instruction.wires)
        elif instruction.name in ("h", "x", "cx", "ccx"):
            result.gate(instruction.name, instruction.wires)
        else:
            raise ValueError(
                "The source compilation path requires RY/RX and native Clifford controls"
            )
    return result, tuple(errors)
