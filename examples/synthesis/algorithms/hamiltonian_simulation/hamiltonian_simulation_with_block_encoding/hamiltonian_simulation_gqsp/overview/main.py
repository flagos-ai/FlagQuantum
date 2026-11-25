"""Fit the degree-33 Laurent polynomial and execute source GQSP evolution."""

from pathlib import Path

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
    ALPHA,
    TIME,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.plots import (
    save_comparison,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.workflow import (
    Projection,
)

from .phases import DEGREE, Factorization, factor
from .workflow import block_projection, evolution_projection

OUTPUT = Path("artifacts/development/gqsp-hamiltonian")


def show_factorization(phases: Factorization) -> None:
    """Report observed coefficient reconstruction and approximation residuals."""
    print(
        "Evolution time, Laurent degree and phase rows:", TIME, DEGREE, len(phases.rows)
    )
    print(f"Completion residual: {phases.completion_residual:.12g}")
    print(f"Stripping residual: {phases.stripping_residual:.12g}")
    print(f"Reconstruction residual: {phases.reconstructed_residual:.12g}")
    print(f"Truncation grid error: {phases.approximation_error:.12g}")


def show_and_save(name: str, label: str, projection: Projection) -> Path:
    """Report absolute native coefficients, scale, and circuit size."""
    print(f"{name} absolute error:", projection.error)
    print(f"{name} raw projected amplitudes:", projection.raw.tolist())
    print(f"{name} rescaling:", projection.scale)
    print(f"{name} qubits:", projection.circuit.n_qubits)
    print(f"{name} IR instructions:", len(projection.circuit.to_ir().instructions))
    return save_comparison(projection, label, OUTPUT / f"{name}.png")


def main() -> None:
    """Execute the original Pauli projection and five-qubit GQSP workflow."""
    phases = factor(TIME * ALPHA)
    show_factorization(phases)
    print(show_and_save("pauli", "Pauli block encoding", block_projection()))
    print(show_and_save("gqsp", "GQSP time evolution", evolution_projection(phases)))


if __name__ == "__main__":
    main()
