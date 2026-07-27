"""Compare random-isometric and dimer-singlet 256q training probes."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def series(path: Path) -> dict:
    data = json.loads(path.read_text())
    records = data["rank_records"]
    result = {"energy": [], "discarded": [], "e2e": [], "forward": []}
    for step in range(int(data["steps"])):
        rows = [record["training"]["step_metrics"][step] for record in records]
        result["energy"].append(float(rows[0]["loss"]))
        result["discarded"].append(sum(float(row["discarded_weight"]) for row in rows))
        result["e2e"].append(max(float(row["end_to_end_seconds"]) for row in rows))
        result["forward"].append(max(float(row["forward_seconds"]) for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--dimer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random, dimer = series(args.random), series(args.dimer)
    steps = list(range(len(random["energy"])))
    theory = 256 * (1 - 4 * __import__("math").log(2))
    colors = {"random": "#9aa0a6", "dimer": "#2468b4"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))

    axes[0, 0].plot(steps, dimer["energy"], "o-", color=colors["dimer"], label="Dimer singlet")
    axes[0, 0].axhline(theory, color="#d24b40", linestyle="--", label=f"Thermodynamic estimate {theory:.1f}")
    axes[0, 0].set_title("(a) Physical energy trajectory", loc="left", weight="bold")
    axes[0, 0].set_ylabel("Total energy")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, random["energy"], "o-", color=colors["random"], label="Random-isometric")
    axes[0, 1].plot(steps, dimer["energy"], "o-", color=colors["dimer"], label="Dimer singlet")
    axes[0, 1].set_title("(b) Initial-state energy scale", loc="left", weight="bold")
    axes[0, 1].set_ylabel("Total energy")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].semilogy(steps, random["discarded"], "o-", color=colors["random"], label="Random-isometric")
    axes[1, 0].semilogy(steps, dimer["discarded"], "o-", color=colors["dimer"], label="Dimer singlet")
    axes[1, 0].set_title("(c) Truncation differs by orders of magnitude", loc="left", weight="bold")
    axes[1, 0].set_ylabel("Total discarded weight (log scale)")
    axes[1, 0].set_xlabel("Adam step")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(steps, random["e2e"], "o-", color=colors["random"], label="Random-isometric")
    axes[1, 1].plot(steps, dimer["e2e"], "o-", color=colors["dimer"], label="Dimer singlet")
    axes[1, 1].set_title("(d) End-to-end runtime tradeoff", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Seconds / training step")
    axes[1, 1].set_xlabel("Adam step")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(steps)
        axis.grid(alpha=0.22)
    fig.suptitle(
        "Initial state changes both physics and execution path\n"
        "256q · p=16 · χ=512 · Adam lr=0.002 · compiled gesvda · 8×A800",
        fontsize=14, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "random_isometric": random,
        "dimer_singlet": dimer,
        "thermodynamic_ground_energy_estimate": theory,
        "conclusion": "dimer_is_physically_stable_but_exposes_unoptimized_near_exact_dynamic_shape_path",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
