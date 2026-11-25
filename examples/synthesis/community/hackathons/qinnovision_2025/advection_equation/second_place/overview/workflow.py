"""Execute the full seven-bit, thirty-two-step advection exercise."""

from dataclasses import dataclass

import torch

import flagquantum as fq

from . import physics, quantum


@dataclass(frozen=True)
class Result:
    axis: tuple[float, ...]
    boundary: tuple[float, ...]
    classical: tuple[float, ...]
    quantum_probability_profile: tuple[float, ...]
    state_error: float
    width: int
    depth: int
    cx: int
    times: tuple[float, ...]
    measured_cx: tuple[int, ...]
    theory_cx: tuple[int, ...]
    cry_widths: tuple[int, ...]
    measured_cry: tuple[int, ...]
    theory_cry: tuple[int, ...]


def theoretical_cx(width: int, time: float, tau: float = 1 / 32) -> int:
    return int(time // tau) * (9 * width * width - 33 * width + 34)


def run() -> Result:
    circuit = quantum.build()
    state = quantum.state(circuit)
    expected = physics.reference(7, 1, 1 / 32)
    error = float((state - expected).abs().max())
    metrics = quantum.metrics(circuit)
    times = (0.5, 1.0, 1.5, 2.0)
    measured = tuple(
        quantum.metrics(quantum.build(time=t)).operation_counts.get("cx", 0)
        for t in times
    )
    widths = tuple(range(2, 8))
    cry = []
    for width in widths:
        test = fq.Circuit(width, dtype=torch.complex128)
        for control in range(width - 1):
            test.h(control)
        test.x(width - 1)
        quantum.controlled_ry(test, tuple(range(width - 1)), width - 1, torch.pi / 4)
        cry.append(quantum.metrics(test).operation_counts.get("cx", 0))
    axis = physics.grid()
    boundary = ((axis > 1) & (axis < 2)).to(torch.float64)
    return Result(
        tuple(float(x) for x in axis),
        tuple(float(x) for x in boundary),
        tuple(float(x) for x in physics.classical()),
        tuple(float(x) for x in 32 * state.abs().square()),
        error,
        metrics.n_qubits,
        metrics.depth,
        metrics.operation_counts.get("cx", 0),
        times,
        measured,
        tuple(theoretical_cx(7, t) for t in times),
        widths,
        tuple(cry),
        tuple(16 * n - 40 if n > 2 else 2 for n in widths),
    )
