#!/usr/bin/env python3
"""Build an auditable SV/MPS/TN backend phase diagram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

import flagquantum as fq


def chain(n: int, depth: int = 4) -> fq.Circuit:
    circuit = fq.Circuit(n)
    for layer in range(depth):
        for wire in range(layer % 2, n - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def tree(n: int) -> fq.Circuit:
    circuit = fq.Circuit(n)
    for child in range(1, n):
        circuit.cx((child - 1) // 2, child)
    return circuit


def grid(rows: int, columns: int, cycles: int = 4) -> fq.Circuit:
    circuit = fq.Circuit(rows * columns)
    for _ in range(cycles):
        for row in range(rows):
            for column in range(columns - 1):
                circuit.cx(row * columns + column, row * columns + column + 1)
        for row in range(rows - 1):
            for column in range(columns):
                circuit.cx(row * columns + column, (row + 1) * columns + column)
    return circuit


def main() -> None:
    cases = []
    for n in (30, 36, 40, 64, 128, 256, 512, 1024):
        cases.extend((("chain", n, chain(n)), ("binary_tree", n, tree(n))))
    for rows, columns in ((5, 6), (6, 6), (8, 8)):
        cases.append(("grid", rows * columns, grid(rows, columns)))
    targets = (
        ("full_state", 1),
        ("samples", 1024),
        ("single_amplitude", 1),
        ("few_amplitudes", 4),
        ("local_observables", 8),
    )
    records = []
    for topology, n, circuit in cases:
        for target, count in targets:
            decision = fq.select_backend_by_cost(
                circuit,
                target=target,
                target_count=count,
                complex_bytes=8,
                memory_limit_bytes=80 << 30,
            )
            records.append(
                {
                    "topology": topology,
                    "qubits": n,
                    "target": target,
                    "target_count": count,
                    **decision.summary(),
                }
            )

    root = Path("benchmarks/results/legacy/backend_phase_diagram_20260730")
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    sparse = [
        item
        for item in records
        if item["target"] == "single_amplitude"
    ]
    colors = {"statevector": "#4c78a8", "mps": "#f58518", "tensor_network": "#54a24b"}
    markers = {"chain": "o", "binary_tree": "^", "grid": "s"}
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    topology_y = {"chain": 0, "binary_tree": 1, "grid": 2}
    for item in sparse:
        ax.scatter(
            item["qubits"],
            topology_y[item["topology"]],
            color=colors[item["selected_backend"]],
            marker=markers[item["topology"]],
            s=90,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks((30, 36, 40, 64, 128, 256, 512, 1024))
    ax.set_xticklabels(("30", "36", "40", "64", "128", "256", "512", "1024"))
    ax.set_yticks((0, 1, 2), ("1D chain", "nonlocal tree", "2D grid"))
    ax.set_xlabel("qubits")
    ax.set_title("Backend phase diagram: one amplitude, 80 GiB/rank")
    for backend, color in colors.items():
        ax.scatter([], [], color=color, label=backend, s=70)
    ax.legend(ncols=3, loc="upper center")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"backend_phase_diagram.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
