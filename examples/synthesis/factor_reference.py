"""Independent Circuit phases from small classical cost-factor truth tables."""

import math
from collections import defaultdict
from collections.abc import Sequence

import torch

import flagquantum as fq

# Table index bit i addresses logical input bits[i], independently of wire layout.
CostFactor = tuple[tuple[int, ...], tuple[float, ...]]


def squared_residual(
    bits: tuple[int, ...],
    coefficients: tuple[float, ...],
    offset: float,
    weight: float,
) -> CostFactor:
    """Enumerate one bounded affine residual, preserving its constant phase."""
    if len(bits) != len(coefficients) or len(bits) > 6:
        raise ValueError("A residual requires matching support of at most six bits")
    values = tuple(
        weight
        * (offset + sum(c * ((basis >> i) & 1) for i, c in enumerate(coefficients)))
        ** 2
        for basis in range(1 << len(bits))
    )
    return bits, values


def factored_cost_circuit(
    width: int,
    factors: Sequence[CostFactor],
    params: tuple[float, ...],
    *,
    mixer_scale: float = 1.0,
) -> fq.Circuit:
    """Build exact cost/RX layers without a full 2**width classical cost table.

    Parameters alternate gamma and beta. Each at-most-eight-bit truth table is
    independently Walsh transformed into spin parities. This is a reference
    implementation using Circuit only, not a synthesis or execution backend.
    """
    if not 1 <= width <= 24 or not params or len(params) % 2:
        raise ValueError("Use 1..24 bits and nonempty alternating gamma/beta values")
    if not all(math.isfinite(x) for x in (*params, mixer_scale)):
        raise ValueError("Reference angles must be finite")
    parities: dict[tuple[int, ...], float] = defaultdict(float)
    for bits, table in factors:
        if (
            len(bits) > 8
            or len(set(bits)) != len(bits)
            or any(not 0 <= bit < width for bit in bits)
            or len(table) != 1 << len(bits)
            or not all(math.isfinite(value) for value in table)
        ):
            raise ValueError(
                "Cost factors require distinct in-bounds bits and finite tables"
            )
        values = list(table)
        for bit in range(len(bits)):
            for mask in range(len(values)):
                if mask & (1 << bit) == 0:
                    first, second = values[mask], values[mask | (1 << bit)]
                    values[mask] = first + second
                    values[mask | (1 << bit)] = first - second
        for mask, value in enumerate(values):
            support = tuple(
                sorted(bits[i] for i in range(len(bits)) if mask & (1 << i))
            )
            parities[support] += value / len(table)
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for layer in range(len(params) // 2):
        gamma, beta = params[2 * layer : 2 * layer + 2]
        for support, coefficient in sorted(parities.items()):
            if coefficient == 0:
                continue
            angle = gamma * coefficient
            wires = tuple(width - 1 - bit for bit in support)
            if not wires:
                phase = torch.exp(torch.tensor(1j * angle, dtype=torch.complex128))
                circuit.any(0, matrix=phase * torch.eye(2, dtype=torch.complex128))
                continue
            target = wires[-1]
            for control in wires[:-1]:
                circuit.cx(control, target)
            circuit.rz(target, theta=-2 * angle)
            for control in reversed(wires[:-1]):
                circuit.cx(control, target)
        for wire in range(width):
            circuit.rx(wire, theta=mixer_scale * beta)
    return circuit
