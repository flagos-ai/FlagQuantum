"""Run the original winning advection notebook with public native circuits."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print(f"Grid: {len(result.axis)} points, seven qubits, original 10-step evolution")
    print("RK4 step: 0.038; original t=1 readout selects index 26, actual time 0.988")
    print(f"Fresh Pauli terms: {result.pauli_terms}")
    print(f"Reassembled Hamiltonian error: {result.decomposition_error:.12g}")
    for curve in result.curves:
        print(
            f"{curve.name}: fidelity={curve.fidelity:.12g}, full absolute state error={curve.absolute_state_error:.12g}, global phase={curve.phase:.12g}, sign={curve.sign:g}, width={curve.width}, CX={curve.cx}, depth={curve.depth}"
        )
    print(f"Qubit sweep: {result.qubit_widths}")
    print(f"Qubit-sweep CX counts (proposed, custom, trotter): {result.qubit_cx}")
    print(f"Original step sweep: {result.step_counts}")
    print(f"Step-sweep CX counts (proposed, custom, trotter): {result.step_cx}")


def write(result: workflow.Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("winning-advection-figures")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("winning-advection-results.json")
    )
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
