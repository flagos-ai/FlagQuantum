"""Measure a native preparation bound sweep on the source's eight-qubit data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc


@dataclass(frozen=True)
class PreparationPoint:
    """Executed preparation and measured errors at one requested upper bound."""

    bound: float
    program: synth.Program
    amplitude_l2_error: float
    probability_l2_error: float
    depth: int
    cx_count: int


@qfunc
def prepare(amplitudes: list[float], bound: float, out: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(amplitudes, bound, out)


def source_amplitudes() -> list[float]:
    """Reproduce the source seed, cubic trend, perturbation and normalization."""
    x = np.linspace(-1, 1, 256)
    trend = 0.6 - 0.4 * x + 0.8 * x**2 - 0.3 * x**3
    amplitudes = trend + 0.05 * np.random.RandomState(1).randn(256)
    amplitudes -= np.mean(amplitudes)
    return [float(value) for value in amplitudes / np.linalg.norm(amplitudes)]


def run_sweep() -> tuple[PreparationPoint, ...]:
    """Execute all ten original bounds and measure actual compiler resources.

    The current exact kernel satisfies every bound; it does not trade accuracy
    for fewer gates. Resource rows use the native instruction basis, not the
    source platform's custom CX/U transpilation basis.
    """
    amplitudes = source_amplitudes()
    expected = torch.tensor(amplitudes, dtype=torch.complex128)
    points = []
    for value in np.linspace(0.0, 0.3, 10):
        bound = float(value)
        program = synth.synthesize(
            prepare,
            classical_inputs={"amplitudes": amplitudes, "bound": bound},
            max_qubits=8,
            max_workspace=0,
            memory_limit_bytes=64 * 1024**2,
        )
        state = fq.run(program).statevector().flatten()
        amplitude_error = float(torch.linalg.vector_norm(state - expected))
        probability_error = float(
            torch.linalg.vector_norm(state.abs().square() - expected.abs().square())
        )
        resources = estimate_resources(program.circuit)
        points.append(
            PreparationPoint(
                bound,
                program,
                amplitude_error,
                probability_error,
                int(resources.depth),
                int(resources.operation_counts.get("cx", 0)),
            )
        )
    return tuple(points)


def plot_sweep(points: tuple[PreparationPoint, ...], output: Path) -> None:
    """Save measured native depth and CX count without claiming compression."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    bounds = [point.bound for point in points]
    axes.plot(bounds, [point.depth for point in points], "o-", label="Native depth")
    axes.plot(
        bounds, [point.cx_count for point in points], "*-", label="Native CX count"
    )
    axes.set_xlabel("Requested amplitude L2 error upper bound")
    axes.set_ylabel("Native depth, CX count")
    axes.set_title("Exact preparation: no approximation savings")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_example(output: Path = Path("native_amplitude_bound_sweep.png")) -> None:
    """Run data generation, all bound experiments, resource reporting and plot."""
    points = run_sweep()
    for point in points:
        print(
            f"bound={point.bound:.6f} depth={point.depth} CX={point.cx_count} "
            f"amplitude_L2={point.amplitude_l2_error:.3g} "
            f"probability_L2={point.probability_l2_error:.3g}"
        )
    plot_sweep(points, output)
    print(f"Saved native resource plot to {output}")


if __name__ == "__main__":
    run_example()
