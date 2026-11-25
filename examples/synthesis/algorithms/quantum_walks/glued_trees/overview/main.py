"""Run both original time sweeps and report their finite exit measurements."""

from pathlib import Path

from .plots import plot_walk
from .quantum import EDGES_6, EXPANSION_ERROR, TERMS_6
from .workflow import exit_word, run_range


def run_example() -> None:
    print("Fresh six-qubit graph edges:", EDGES_6, "Pauli terms:", len(TERMS_6))
    print("Six-qubit Hamiltonian reconstruction error:", EXPANSION_ERROR)
    for qubits in (6, 20):
        print("Running", qubits, "qubits with exit word", exit_word(qubits), flush=True)
        rows = run_range(qubits)
        for row in rows:
            print(row)
        path = Path("artifacts/development/glued-trees") / f"exit-{qubits}.png"
        print(plot_walk(qubits, rows, path))


if __name__ == "__main__":
    run_example()
