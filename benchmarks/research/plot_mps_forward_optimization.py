"""Plot measured distributed-MPS speed/capacity operating points."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def metrics(path: Path, steady: bool):
    data = json.loads(path.read_text())
    count = int(data["steps"])
    forward = []
    e2e = []
    for step in range(count):
        rows = [record["training"]["step_metrics"][step] for record in data["rank_records"]]
        forward.append(max(row["forward_seconds"] for row in rows))
        e2e.append(max(row["end_to_end_seconds"] for row in rows))
    selected = slice(1, None) if steady and count > 1 else slice(None)
    return {
        "forward": statistics.median(forward[selected]),
        "e2e": statistics.median(e2e[selected]),
        "forward_steps": forward,
        "e2e_steps": e2e,
        "memory": max(data["local_memory_bytes_by_rank"]) / 2**30,
        "energy": data["best_variational_energy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--prefetch", type=Path, required=True)
    parser.add_argument("--fast-approx", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eager = metrics(args.eager, False)
    compiled = metrics(args.compiled, True)
    prefetch = metrics(args.prefetch, True)
    fast_approx = metrics(args.fast_approx, True)
    capacity = metrics(args.capacity, True)
    rows = [eager, compiled, prefetch, fast_approx, capacity]
    labels = [
        "Eager\n(equal)",
        "Compiled\n(equal)",
        "+ halo\nprefetch",
        "+ gesvda\n(approx.)",
        "Cost-aware\n(capacity)",
    ]
    colors = ["#9aa0a6", "#57a0d3", "#238b45", "#e28e2c", "#2468b4"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    bars = axes[0].bar(labels, [row["forward"] for row in rows], color=colors)
    axes[0].set_ylabel("Forward time (s)")
    axes[0].set_title("(a) Performance operating point", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].bar_label(bars, fmt="%.2f s", padding=3, fontsize=9)
    exact_speedup = eager["forward"] / prefetch["forward"]
    fast_speedup = eager["forward"] / fast_approx["forward"]
    axes[0].text(0.5, 0.84, f"{exact_speedup:.2f}× exact · {fast_speedup:.2f}× approximate", transform=axes[0].transAxes, ha="center", color="#238b45", weight="bold", fontsize=10.5)

    memory_bars = axes[1].bar(labels, [row["memory"] for row in rows], color=colors)
    axes[1].set_ylabel("Peak allocated memory / rank (GiB)")
    axes[1].set_title("(b) Capacity operating point", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].bar_label(memory_bars, fmt="%.3f", padding=3, fontsize=9)
    reduction = 100 * (1 - capacity["memory"] / eager["memory"])
    axes[1].text(0.5, 0.68, f"−{reduction:.1f}% peak memory", transform=axes[1].transAxes, ha="center", color="#2468b4", weight="bold", fontsize=11)
    fig.suptitle("Distributed MPS optimization · N=32 · p=2 · χ=512 · 8×A800\ncompiled values are medians after the first warmup step", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    exact_energies = [eager["energy"], compiled["energy"], prefetch["energy"], capacity["energy"]]
    args.output.with_suffix(".json").write_text(json.dumps({
        "eager_equal": eager,
        "compiled_equal": compiled,
        "compiled_halo_prefetch": prefetch,
        "compiled_halo_gesvda_approximate": fast_approx,
        "compiled_cost_aware": capacity,
        "exact_steady_forward_speedup": exact_speedup,
        "approximate_steady_forward_speedup": fast_speedup,
        "halo_prefetch_incremental_speedup": compiled["forward"] / prefetch["forward"],
        "gesvda_incremental_speedup": prefetch["forward"] / fast_approx["forward"],
        "capacity_memory_reduction_percent": reduction,
        "maximum_exact_mode_energy_abs_difference": max(exact_energies) - min(exact_energies),
        "approximate_vs_exact_energy_abs_difference": abs(fast_approx["energy"] - prefetch["energy"]),
        "approximate_mode_notice": "gesvda is a fast approximate SVD mode; gesvd remains the default exact mode",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
