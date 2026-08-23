#!/usr/bin/env python3
"""Build the gradient-specific SV/MPS/TN phase diagram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

import flagquantum as fq


def chain(n: int) -> fq.Circuit:
    circuit = fq.Circuit(n)
    for layer in range(4):
        for wire in range(layer % 2, n - 1, 2):
            circuit.ry(wire, theta=0.01 * (layer + 1))
            circuit.cx(wire, wire + 1)
    return circuit


def tree(n: int) -> fq.Circuit:
    circuit = fq.Circuit(n)
    for child in range(1, n):
        circuit.ry(child, theta=0.01)
        circuit.cx((child - 1) // 2, child)
    return circuit


def grid(rows: int, columns: int) -> fq.Circuit:
    circuit = fq.Circuit(rows * columns)
    for cycle in range(4):
        for wire in range(rows * columns):
            circuit.ry(wire, theta=0.01 * (cycle + 1))
        for row in range(rows):
            for column in range(columns - 1):
                circuit.cx(row * columns + column, row * columns + column + 1)
        for row in range(rows - 1):
            for column in range(columns):
                circuit.cx(row * columns + column, (row + 1) * columns + column)
    return circuit


def main() -> None:
    cases = []
    for n in (24, 30, 36, 40, 64, 128, 256):
        cases.extend((("chain", n, chain(n)), ("binary_tree", n, tree(n))))
    for rows, columns in ((4, 6), (5, 6), (6, 6), (8, 8)):
        cases.append(("grid", rows * columns, grid(rows, columns)))
    targets = (("expectation", 1), ("local_observables", 8))
    records = []
    for topology, n, circuit in cases:
        for target, count in targets:
            decision = fq.select_backend_by_cost(
                circuit,
                target=target,
                target_count=count,
                require_gradients=True,
                complex_bytes=16,
                memory_limit_bytes=80 << 30,
            )
            records.append(
                {
                    "topology": topology,
                    "qubits": n,
                    "target": target,
                    "target_count": count,
                    "evidence_level": (
                        "measured_small_fitting_gradient"
                        if n <= 12
                        else "structural_capacity_prediction"
                    ),
                    **decision.summary(),
                }
            )
    root = Path("benchmarks/results/gradient_backend_phase_diagram_20260730")
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    selected = [item for item in records if item["target"] == "expectation"]
    colors = {"statevector": "#4c78a8", "mps": "#f58518", "tensor_network": "#54a24b"}
    y = {"chain": 0, "binary_tree": 1, "grid": 2}
    markers = {"chain": "o", "binary_tree": "^", "grid": "s"}
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for item in selected:
        ax.scatter(
            item["qubits"],
            y[item["topology"]],
            color=colors[item["selected_backend"]],
            marker=markers[item["topology"]],
            s=90,
        )
    ax.set_xscale("log", base=2)
    ticks = (24, 30, 36, 40, 64, 128, 256)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_yticks((0, 1, 2), ("1D chain", "nonlocal tree", "2D grid"))
    ax.set_xlabel("qubits")
    ax.set_title("Gradient backend phase diagram: expectation, complex128")
    for backend, color in colors.items():
        ax.scatter([], [], color=color, label=backend, s=70)
    ax.legend(ncols=3, loc="upper center")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"gradient_backend_phase_diagram.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
