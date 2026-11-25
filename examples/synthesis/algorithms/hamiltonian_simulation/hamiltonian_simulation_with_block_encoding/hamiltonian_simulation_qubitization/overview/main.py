"""Run the original-sized Pauli block encoding and qubitized time evolution."""

from pathlib import Path

from .model import ALPHA, DEGREE, TIME, expansion, fresh_input
from .plots import save_comparison
from .workflow import run_workflow

FOLDER = Path("artifacts/development/qubitized-hamiltonian")


def main() -> None:
    hamiltonian, evolution = run_workflow()
    print("Time and Chebyshev degree:", TIME, DEGREE)
    print("Scales:", ALPHA, expansion().scale)
    print("Fresh input:", fresh_input().tolist())
    for name, label, result in (
        ("pauli", "Pauli block encoding", hamiltonian),
        ("evolution", "Qubitized time evolution", evolution),
    ):
        print(f"{name} error:", result.error)
        print(f"{name} width:", result.circuit.n_qubits)
        print(f"{name} IR instructions:", len(result.circuit.to_ir().instructions))
        print(f"{name} scaled amplitudes:", (result.raw * result.scale).tolist())
        print(save_comparison(result, label, FOLDER / f"{name}.png"))


if __name__ == "__main__":
    main()
