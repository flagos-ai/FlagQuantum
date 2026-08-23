"""Plot measured MPS versus general-TN sparse-output crossover evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((args.results / "raw").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        times = payload["times_seconds"]
        rows.append(
            {
                "topology": payload["topology"],
                "backend": payload["backend"],
                "qubits": payload["qubits"],
                "depth": payload["depth"],
                "targets": payload["target_count"],
                "cold_start_seconds": payload.get("cold_start_seconds", times[0]),
                "steady_state_seconds": payload.get(
                    "steady_state_seconds",
                    statistics.median(times[1:]) if len(times) > 1 else times[0],
                ),
                "peak_memory_mib": payload["peak_memory_bytes"] / 2**20,
                "observed_max_bond": payload["observed_max_bond"] or 0,
                "source": path.name,
            }
        )
    rows.sort(key=lambda row: (row["topology"], row["qubits"], row["backend"]))
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    tree = [row for row in rows if row["topology"] == "binary_tree"]
    colors = {"mps": "#d97706", "tensor_network": "#2563eb"}
    labels = {"mps": "MPS", "tensor_network": "General TN"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for backend in ("mps", "tensor_network"):
        items = [row for row in tree if row["backend"] == backend]
        qubits = [row["qubits"] for row in items]
        axes[0, 0].plot(
            qubits,
            [row["cold_start_seconds"] for row in items],
            "o-",
            color=colors[backend],
            label=labels[backend],
        )
        axes[0, 1].plot(
            qubits,
            [row["steady_state_seconds"] for row in items],
            "o-",
            color=colors[backend],
            label=labels[backend],
        )
        axes[1, 0].plot(
            qubits,
            [row["peak_memory_mib"] for row in items],
            "o-",
            color=colors[backend],
            label=labels[backend],
        )
    mps_tree = [row for row in tree if row["backend"] == "mps"]
    axes[1, 1].plot(
        [row["qubits"] for row in mps_tree],
        [row["observed_max_bond"] for row in mps_tree],
        "o-",
        color=colors["mps"],
        label="MPS observed bond",
    )
    axes[0, 0].set(title="Binary tree · cold start", ylabel="seconds")
    axes[0, 1].set(title="Binary tree · steady state", ylabel="seconds", yscale="log")
    axes[1, 0].set(title="Binary tree · peak memory", ylabel="MiB", yscale="log")
    axes[1, 1].set(
        title="Binary tree · observed MPS bond growth",
        xlabel="qubits",
        ylabel="maximum bond dimension",
        yscale="log",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[0, :]:
        axis.set_xlabel("qubits")
    axes[1, 0].set_xlabel("qubits")
    fig.suptitle("FlagQuantum MPS–TN sparse-output crossover")
    fig.savefig(args.output / "tn_mps_crossover.png", dpi=180)
    fig.savefig(args.output / "tn_mps_crossover.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
