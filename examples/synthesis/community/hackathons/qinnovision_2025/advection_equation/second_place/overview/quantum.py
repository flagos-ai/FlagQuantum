"""Compile the original Dirichlet advection ladder with no work qubits."""

import math
from collections.abc import Sequence
from typing import cast

import torch

import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)

BASIS = ("u3", "cx")


def controlled_ry(
    circuit: fq.Circuit, controls: Sequence[int], target: int, angle: float
) -> None:
    """Exact Gray-code multiplexor for a positive-control real rotation."""
    if not controls:
        circuit.ry(target, theta=angle)
        return
    size = 1 << len(controls)
    for i in range(size):
        gray = i ^ (i >> 1)
        circuit.ry(target, theta=((-1) ** gray.bit_count()) * angle / size)
        next_index = (i + 1) % size
        next_gray = next_index ^ (next_index >> 1)
        changed = (gray ^ next_gray).bit_length() - 1
        circuit.cx(controls[changed], target)


def initialize(circuit: fq.Circuit, width: int) -> None:
    """Exact original plateau on indices N/4 through N/2-1."""
    circuit.x(1)
    for bit in range(2, width):
        circuit.h(bit)


def ladder(circuit: fq.Circuit, width: int, dt: float) -> None:
    # Qmod's QArray element zero is the least significant physical state bit.
    wires = tuple(reversed(range(width)))
    circuit.ry(wires[0], theta=-dt)
    circuit.ry(wires[0], theta=2 * dt)
    for j in range(1, width):
        target = wires[j]
        for control in wires[:j]:
            circuit.cx(target, control)
        controlled_ry(circuit, wires[:j], target, 2 * dt)
        for control in reversed(wires[:j]):
            circuit.cx(target, control)
    circuit.ry(wires[0], theta=dt)


def build(width: int = 7, time: float = 1, tau: float = 1 / 32) -> fq.Circuit:
    if type(width) is not int or not 2 <= width <= 7:
        raise ValueError("keep two through seven spatial bits")
    if not math.isfinite(time) or not math.isfinite(tau) or time <= 0 or tau <= 0:
        raise ValueError("time and tau must be positive finite reals")
    steps = math.floor(time / tau)
    if not 1 <= steps <= 64:
        raise ValueError("the source time sweep needs at most sixty-four steps")
    circuit = fq.Circuit(width, dtype=torch.complex128)
    initialize(circuit, width)
    dt = time * (2**width // 4) / (2 * steps)
    for _ in range(steps):
        ladder(circuit, width, dt)
    return circuit


def state(circuit: fq.Circuit) -> torch.Tensor:
    return cast(torch.Tensor, circuit.state()).reshape(-1)


def metrics(circuit: fq.Circuit) -> ResourceEstimate:
    lowered = convert_basis(circuit, gates=BASIS, max_added_operations=300_000).program
    return estimate_resources(lowered)
