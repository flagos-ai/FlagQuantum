"""Plot the bounded 256-site MPS-VQE convergence diagnostic."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = data["rank_records"]
    steps = list(range(int(data["steps"])))
    energy, grad_l2, grad_max, discarded, forward, reverse, e2e = ([] for _ in range(7))
    for step in steps:
        rows = [record["training"]["step_metrics"][step] for record in records]
        gradients = {}
        for row in rows:
            gradients.update({int(index): float(value) for index, value in row["parameter_gradients"]})
        values = list(gradients.values())
        energy.append(float(rows[0]["loss"]))
        grad_l2.append(math.sqrt(sum(value * value for value in values)))
        grad_max.append(max(map(abs, values)))
        discarded.append(sum(float(row["discarded_weight"]) for row in rows))
        forward.append(max(float(row["forward_seconds"]) for row in rows))
        reverse.append(max(float(row["reverse_seconds"]) for row in rows))
        e2e.append(max(float(row["end_to_end_seconds"]) for row in rows))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    best = min(range(len(energy)), key=energy.__getitem__)
    axes[0, 0].plot(steps, energy, "o-", color="#2468b4", linewidth=2)
    axes[0, 0].scatter([best], [energy[best]], s=75, color="#d24b40", zorder=3)
    axes[0, 0].annotate(
        f"best at step {best}: {energy[best]:.6f}",
        (best, energy[best]),
        xytext=(25, -28),
        textcoords="offset points",
        color="#d24b40",
        arrowprops={"arrowstyle": "->", "color": "#d24b40"},
    )
    axes[0, 0].set_title("(a) Energy: decreases, then rebounds", loc="left", weight="bold")
    axes[0, 0].set_ylabel("Energy")

    axes[0, 1].plot(steps, grad_l2, "o-", label="Gradient L2", color="#238b45", linewidth=2)
    axes[0, 1].plot(steps, grad_max, "s--", label="Max |gradient|", color="#57a0d3")
    axes[0, 1].set_title("(b) Gradient remains non-zero", loc="left", weight="bold")
    axes[0, 1].set_ylabel("Gradient magnitude")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(steps, discarded, "o-", color="#d98c10", linewidth=2)
    axes[1, 0].set_title("(c) Truncation remains extreme", loc="left", weight="bold")
    axes[1, 0].set_ylabel("Total discarded weight")
    axes[1, 0].set_xlabel("Optimizer step")

    axes[1, 1].plot(steps, e2e, "o-", label="End-to-end", color="#6a3d9a", linewidth=2)
    axes[1, 1].plot(steps, forward, "s--", label="Forward", color="#2468b4")
    axes[1, 1].plot(steps, reverse, "^--", label="Reverse", color="#ef8a62")
    axes[1, 1].set_title("(d) Runtime after cold step", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_xticks(steps)
    fig.suptitle(
        "256q MPS-VQE convergence probe · p=16 · χ=512 · Adam · 8×A800\n"
        "compiled approximate gesvda · random-isometric initial state",
        fontsize=14, weight="bold",
    )
    fig.text(0.99, 0.01, "Capacity/performance stress test; not a certified physics-accuracy trajectory.", ha="right", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "energy": energy, "gradient_l2": grad_l2, "gradient_max_abs": grad_max,
        "total_discarded_weight": discarded, "forward_seconds": forward,
        "reverse_seconds": reverse, "end_to_end_seconds": e2e,
        "best_step": best, "best_energy": energy[best],
        "conclusion": "not_converged_energy_rebounded_after_step_5",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
