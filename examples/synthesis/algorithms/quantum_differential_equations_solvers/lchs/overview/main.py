"""Run the complete classical and quantum LCHS workflow locally."""

from pathlib import Path

from .model import GRID_BITS, Grid, source_grid, source_input
from .plots import save_kernel
from .workflow import (
    CLASSICAL_SIZE,
    QuantumComparison,
    classical_reference,
    compare_quantum,
)

OUTPUT = Path("artifacts/development/lchs/kernel.png")


def show_grid(grid: Grid) -> None:
    """Print actual derived source grid parameters."""
    print("Grid bits and points:", GRID_BITS, len(grid.signed))
    print("Gamma, radius and step:", grid.gamma, grid.radius, grid.step)


def show_quantum(result: QuantumComparison) -> None:
    """Report the complete native projection and comparison measurements."""
    degree = (len(result.factorization.rows) - 1) // 2
    print("Laurent degree and phases:", degree, len(result.factorization.rows))
    print("Actual logical qubits:", result.circuit.n_qubits)
    print("Native IR instructions:", len(result.circuit.to_ir().instructions))
    print("Absolute quantum/quadrature error:", result.absolute_error)
    print("Postselected complex amplitudes:", result.amplitudes.tolist())
    print("Quantum/quadrature fidelity:", result.quadrature_fidelity)
    print("Quantum/exact fidelity:", result.exact_fidelity)
    print("Postselection amplitude norm:", float(result.amplitudes.norm()))


def main() -> None:
    """Regenerate the original 128-dimensional and 12-qubit workflows."""
    grid = source_grid()
    show_grid(grid)
    print(
        "Classical dimension and fidelity:", CLASSICAL_SIZE, classical_reference(grid)
    )
    print(save_kernel(grid, OUTPUT))
    show_quantum(compare_quantum(grid, source_input(4)))


if __name__ == "__main__":
    main()
