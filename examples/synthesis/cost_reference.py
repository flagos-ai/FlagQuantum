"""Independent small-system Circuit references for layered classical costs."""

from collections.abc import Callable

import torch

import flagquantum as fq


def cost_phase_circuit(
    cost: Callable[[tuple[int, ...]], float],
    width: int,
    params: tuple[float, ...],
) -> fq.Circuit:
    """Apply cost phases and RX mixers using a bounded spin-parity expansion.

    This reference evaluates all basis costs independently of synthesis and uses
    O(2**width) classical memory. Gamma values precede beta values in params.
    """
    if not 1 <= width <= 10 or not params or len(params) % 2:
        raise ValueError("Reference requires 1..10 bits and paired phase/mixer values")
    coefficients = [
        cost(tuple((basis >> i) & 1 for i in range(width))) for basis in range(2**width)
    ]
    # Walsh transform: spin parity, rather than the compiler's Boolean monomials.
    for bit in range(width):
        for mask in range(2**width):
            if mask & (1 << bit) == 0:
                first, second = coefficients[mask], coefficients[mask | (1 << bit)]
                coefficients[mask] = first + second
                coefficients[mask | (1 << bit)] = first - second
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    layers = len(params) // 2
    for layer in range(layers):
        for mask, value in enumerate(coefficients):
            if value == 0:
                continue
            angle = params[layer] * value / 2**width
            wires = [wire for wire in range(width) if mask & (1 << (width - 1 - wire))]
            if not wires:
                matrix = torch.exp(torch.tensor(1j * angle, dtype=torch.complex128))
                circuit.any(0, matrix=matrix * torch.eye(2, dtype=torch.complex128))
                continue
            target = wires[-1]
            for control in wires[:-1]:
                circuit.cx(control, target)
            circuit.rz(target, theta=-2 * angle)
            for control in reversed(wires[:-1]):
                circuit.cx(control, target)
        for wire in range(width):
            circuit.rx(wire, theta=params[layers + layer])
    return circuit
