"""Aggregate and plot measured sparse-output TN capacity evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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
        work = payload["workload"]
        rows.append(
            {
                "qubits": work["qubits"],
                "depth": work["depth"],
                "targets": work["target_count"],
                "world_size": payload["distributed"]["world_size"],
                "latency_seconds": payload["latency_seconds_max_rank_median"],
                "peak_memory_gib": payload["peak_memory_bytes_max"] / 2**30,
                "slice_tasks": payload["execution_summary"]["slice_tasks"],
                "reduction_bytes": payload["execution_summary"][
                    "reduction_payload_bytes"
                ],
                "source": path.name,
            }
        )
    rows.sort(key=lambda row: (row["qubits"], row["world_size"]))
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qubits"]].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    scaling = max(grouped.values(), key=len)
    worlds = [item["world_size"] for item in scaling]
    latency = [item["latency_seconds"] for item in scaling]
    speedup = [latency[0] / value for value in latency]
    efficiency = [value / world * 100 for value, world in zip(speedup, worlds)]
    axes[0, 0].plot(worlds, latency, "o-", label=f"{scaling[0]['qubits']}q measured")
    axes[0, 1].plot(worlds, speedup, "o-", label="measured")
    axes[0, 1].plot(worlds, worlds, "--", color="gray", label="ideal")
    axes[1, 0].plot(worlds, efficiency, "o-", color="#d97706", label="measured")
    capacity = sorted(
        (row for row in rows if row["world_size"] == max(worlds)),
        key=lambda row: row["qubits"],
    )
    capacity_qubits = [row["qubits"] for row in capacity]
    capacity_latency = [row["latency_seconds"] for row in capacity]
    capacity_memory = [row["peak_memory_gib"] for row in capacity]
    axes[1, 1].plot(
        capacity_qubits,
        capacity_latency,
        "o-",
        color="#2563eb",
        label="latency",
    )
    memory_axis = axes[1, 1].twinx()
    memory_axis.plot(
        capacity_qubits,
        capacity_memory,
        "s--",
        color="#16a34a",
        label="memory/rank",
    )
    memory_axis.set_ylabel("peak allocated GiB/rank", color="#16a34a")
    max_world = max(worlds)
    axes[0, 0].set(title="End-to-end latency", ylabel="seconds")
    axes[0, 1].set(title="Strong-scaling speedup", ylabel="speedup")
    axes[1, 0].set(title="Parallel efficiency", ylabel="percent")
    axes[1, 1].set(
        title=f"Capacity sweep on {max_world} A800 GPUs",
        xlabel="qubits",
        ylabel="seconds",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("A800 GPUs")
        axis.set_xticks(worlds)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="upper left")
    memory_axis.legend(loc="lower right")
    fig.suptitle("FlagQuantum sparse-output TN capacity and scaling")
    fig.savefig(args.output / "tn_sparse_capacity.png", dpi=180)
    fig.savefig(args.output / "tn_sparse_capacity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
