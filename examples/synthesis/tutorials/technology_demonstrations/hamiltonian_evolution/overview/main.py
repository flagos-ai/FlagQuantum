"""Benchmark the original water Pauli evolution schedules in a common CX/U basis."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .benchmark import (
    HISTORICAL_QISKIT_CX,
    HISTORICAL_QISKIT_DEPTH,
    Model,
    Result,
    fresh_model,
    run_qdrift,
    run_trotter,
)


def print_model(model: Model) -> None:
    print("Fresh Pauli terms:", len(model.terms))
    print("Fresh RHF energy:", model.rhf_energy)
    print("Maximum source coefficient difference:", model.coefficient_error)


def print_results(rows: tuple[Result, ...]) -> None:
    print("Labels:", tuple(row.label for row in rows))
    print("CX counts:", tuple(row.target.operation_counts.get("cx", 0) for row in rows))
    print("Target depths:", tuple(row.target.depth for row in rows))
    print("Physical qubits:", tuple(row.target.n_qubits for row in rows))
    print(
        "Maximum independent amplitude error:", max(row.reference_error for row in rows)
    )
    print(
        "Maximum converted amplitude error:", max(row.conversion_error for row in rows)
    )
    print("Historical Qiskit CX counts:", HISTORICAL_QISKIT_CX)
    print("Historical Qiskit depths:", HISTORICAL_QISKIT_DEPTH)


def save_figure(rows: tuple[Result, ...], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 1, figsize=(9.6, 7), sharex=True, layout="constrained"
    )
    labels = [row.label for row in rows]
    for axis, title, observed, historical in (
        (
            axes[0],
            "CX count",
            [row.target.operation_counts.get("cx", 0) for row in rows],
            HISTORICAL_QISKIT_CX,
        ),
        (
            axes[1],
            "Circuit depth",
            [row.target.depth for row in rows],
            HISTORICAL_QISKIT_DEPTH,
        ),
    ):
        axis.semilogy(
            labels,
            observed,
            "o-",
            label="FlagQuantum, measured locally",
            color="#236d94",
        )
        axis.semilogy(
            labels, historical, "s--", label="Qiskit, source-reported", color="#9d563c"
        )
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    axes[1].tick_params(axis="x", labelrotation=23)
    figure.suptitle("Water Hamiltonian evolution in the CX/U basis")
    figure.savefig(directory / "evolution-cx-depth.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("hamiltonian-evolution-figures")
    )
    args = parser.parse_args()
    model = fresh_model()
    print_model(model)
    trotter, controlled = run_trotter(model)
    rows = (*trotter, *controlled, *run_qdrift(model))
    print_results(rows)
    save_figure(rows, args.figures)


if __name__ == "__main__":
    main()
