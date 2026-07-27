"""Plot measured equal-site versus cost-aware distributed MPS partitioning."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    data = json.loads(path.read_text())
    steps = [record["training"]["step_metrics"][0] for record in data["rank_records"]]
    return {
        "label": "Equal sites" if data["partition_policy"] == "equal_sites" else "Cost-aware",
        "forward": max(step["forward_seconds"] for step in steps),
        "e2e": max(step["end_to_end_seconds"] for step in steps),
        "memory": max(data["local_memory_bytes_by_rank"]) / 2**30,
        "energy": data["best_variational_energy"],
        "ownership": data["rank_ownership"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal", type=Path, required=True)
    parser.add_argument("--cost-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [load(args.equal), load(args.cost_aware)]
    labels = [row["label"] for row in rows]
    colors = ["#9aa0a6", "#2468b4"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    width = 0.34
    x = range(len(rows))
    axes[0].bar([v - width / 2 for v in x], [r["forward"] for r in rows], width, label="Forward", color="#d98c10")
    axes[0].bar([v + width / 2 for v in x], [r["e2e"] for r in rows], width, label="Full step", color="#238b45")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("(a) Time: no measurable improvement", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=9)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    bars = axes[1].bar(labels, [row["memory"] for row in rows], color=colors)
    axes[1].set_ylabel("Peak allocated memory / rank (GiB)")
    axes[1].set_title("(b) Capacity: lower peak memory", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].bar_label(bars, fmt="%.3f GiB", fontsize=9, padding=3)
    reduction = 100 * (1 - rows[1]["memory"] / rows[0]["memory"])
    axes[1].text(0.5, 0.82, f"−{reduction:.1f}% peak memory", transform=axes[1].transAxes, ha="center", fontsize=11, weight="bold", color="#2468b4")
    fig.suptitle("Cost-aware MPS partitioning · N=32 · p=2 · χ=512 · 8×A800\nenergy exactly aligned in the measured A/B", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({"equal": rows[0], "cost_aware": rows[1], "peak_memory_reduction_percent": reduction, "energy_abs_difference": abs(rows[0]["energy"] - rows[1]["energy"])}, indent=2) + "\n")


if __name__ == "__main__":
    main()
