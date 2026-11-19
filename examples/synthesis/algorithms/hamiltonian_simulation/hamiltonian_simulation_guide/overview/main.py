"""Compare three two-qubit evolution methods over the complete source time grid."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.runtime import ExecutionResult
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    qdrift,
    qfunc,
    suzuki_trotter,
    unitary,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

HAMILTONIAN = 0.3 * (fq.Z(0) @ fq.Z(1)) + 0.7 * fq.X(0) + 0.2 * fq.X(1)
METHODS = ("Suzuki-Trotter 1st order", "Exponentiate", "qDRIFT")
TIME_POINTS = tuple(float(t) for t in torch.linspace(0, 2, 100, dtype=torch.float64))
X_MATRIX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Z_MATRIX = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
IDENTITY = torch.eye(2, dtype=torch.complex128)
HAMILTONIAN_MATRIX = (
    0.3 * torch.kron(Z_MATRIX, Z_MATRIX)
    + 0.7 * torch.kron(IDENTITY, X_MATRIX)
    + 0.2 * torch.kron(X_MATRIX, IDENTITY)
)
EIGENVALUES, EIGENVECTORS = torch.linalg.eigh(HAMILTONIAN_MATRIX)
MAGNETIZATION_DIAGONAL = torch.tensor([1, 0, 0, -1], dtype=torch.float64)


@qfunc
def initial_trotter(qba: Output[QArray[QBit]]) -> None:
    allocate(2, qba)
    suzuki_trotter(HAMILTONIAN, 1.0, 1, 10, qba)


@qfunc
def trotter(t: float, qba: Output[QArray[QBit]]) -> None:
    allocate(2, qba)
    suzuki_trotter(HAMILTONIAN, t, 1, 30, qba)


@qfunc
def drift(t: float, qba: Output[QArray[QBit]]) -> None:
    allocate(2, qba)
    qdrift(HAMILTONIAN, t, 288, qba, 0)


def exact_program(t: float) -> synth.Program:
    """Specialize a static native unitary, preserving its absolute complex phase."""
    evolution = (EIGENVECTORS * torch.exp(-1j * t * EIGENVALUES)) @ EIGENVECTORS.mH
    matrix = tuple(tuple(complex(value) for value in row) for row in evolution.tolist())

    @qfunc
    def exact(qba: Output[QArray[QBit]]) -> None:
        allocate(2, qba)
        unitary(matrix, qba)

    return synth.synthesize(exact, max_workspace=0)


def initial_programs() -> dict[str, synth.Program]:
    return {
        METHODS[0]: synth.synthesize(initial_trotter, max_workspace=0),
        METHODS[1]: exact_program(1.0),
        METHODS[2]: synth.synthesize(
            drift, classical_inputs={"t": 1.0}, max_workspace=0
        ),
    }


def magnetization(program: synth.Program, result: ExecutionResult) -> float:
    """Evaluate the diagonal observable from decoded, named logical values."""
    diagonal = (1.0, 0.0, 0.0, -1.0)
    return math.fsum(
        diagonal[value] * probability
        for (value,), probability in program.probabilities(
            result, registers=("qba",)
        ).items()
    )


def run_sweep(
    times: Sequence[float],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[float]]]:
    states: dict[str, list[torch.Tensor]] = {name: [] for name in METHODS}
    curves: dict[str, list[float]] = {name: [] for name in METHODS}
    for t in times:
        programs = {
            METHODS[0]: synth.synthesize(
                trotter, classical_inputs={"t": t}, max_workspace=0
            ),
            METHODS[1]: exact_program(t),
            METHODS[2]: synth.synthesize(
                drift, classical_inputs={"t": t}, max_workspace=0
            ),
        }
        for name, program in programs.items():
            result = fq.run(program)
            states[name].append(result.statevector().flatten())
            curves[name].append(magnetization(program, result))
    return states, curves


def ideal_curve(times: Sequence[float]) -> list[float]:
    values = []
    for t in times:
        state = torch.matrix_exp(-1j * t * HAMILTONIAN_MATRIX)[:, 0]
        values.append(float((state.abs().square() * MAGNETIZATION_DIAGONAL).sum()))
    return values


def plot_results(
    times: Sequence[float], curves: dict[str, list[float]], ideal: Sequence[float]
) -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.set_xlabel("Evolution coefficient")
    axes.set_ylabel("Magnetization")
    for name in METHODS:
        axes.plot(times, curves[name], label=name)
    axes.plot(times, ideal, label="Ideal Simulation")
    axes.legend()
    plt.show()
    return figure


def run_example() -> None:
    for name, program in initial_programs().items():
        print("Initial method:", name)
        print("Resources:", program.report["resources"])
    states, curves = run_sweep(TIME_POINTS)
    print("Time points:", len(TIME_POINTS))
    print("Sweep states:", sum(len(values) for values in states.values()))
    plot_results(TIME_POINTS, curves, ideal_curve(TIME_POINTS))


if __name__ == "__main__":
    run_example()
