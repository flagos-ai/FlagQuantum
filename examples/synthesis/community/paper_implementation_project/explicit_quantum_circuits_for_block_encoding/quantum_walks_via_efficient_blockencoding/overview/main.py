"""Run the source graph and structured walk experiments."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import LOCAL_SHOTS, REPETITIONS, Result, run


def print_report(result: Result) -> None:
    print("Tree sizes:", tuple(row.size for row in result.trees))
    print("Pauli term counts:", tuple(row.lcu_terms for row in result.trees))
    print(
        "Oracle widths and depths:",
        tuple(
            (row.oracle_resources.n_qubits, row.oracle_resources.depth)
            for row in result.trees
        ),
    )
    print(
        "Pauli LCU widths and depths:",
        tuple(
            (row.lcu_resources.n_qubits, row.lcu_resources.depth)
            for row in result.trees
        ),
    )
    print(
        "Maximum tree oracle and LCU errors:",
        max(row.oracle_error for row in result.trees),
        max(row.lcu_error for row in result.trees),
    )
    print(
        "Circulant and Chebyshev maximum errors:",
        result.cyclic_error,
        result.chebyshev_error,
    )
    print(
        "Tree eight-vertex projected state:", result.trees[2].projection.real.tolist()
    )
    print("Circulant projected state:", result.cyclic_projection.real.tolist())
    print("Chebyshev projected state:", result.chebyshev_projection.real.tolist())
    print(
        "Maximum discriminant block error:",
        max(row.block_error for row in result.walks),
    )
    print(
        "Maximum complete walk state error:",
        max(row.state_error for row in result.walks),
    )
    print("Repetitions and draws per point:", len(REPETITIONS), LOCAL_SHOTS)
    print("Total walk draws:", len(result.walks) * len(REPETITIONS) * LOCAL_SHOTS)
    for walk in result.walks:
        print(
            ("Marked" if walk.marked else "Unmarked") + " observed block-zero counts:",
            walk.counts,
        )


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    sizes = [row.size for row in result.trees]
    for axis, attribute, ylabel in (
        (axes[0], "depth", "CX/U depth"),
        (axes[1], "n_qubits", "Physical qubits"),
    ):
        axis.plot(
            sizes,
            [getattr(row.lcu_resources, attribute) for row in result.trees],
            "s-",
            label="Pauli LCU",
        )
        axis.plot(
            sizes,
            [getattr(row.oracle_resources, attribute) for row in result.trees],
            "o--",
            label="Structured oracle",
        )
        axis.set(xlabel="Vertices", ylabel=ylabel)
        axis.set_xticks(sizes)
        axis.legend()
        axis.grid(alpha=0.25)
    if min(row.lcu_resources.depth for row in result.trees) > 0:
        axes[0].set_yscale("log")
    figure.suptitle("Extended binary tree block encoding")
    figure.savefig(directory / "tree-resources.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for row, color in zip(result.walks, ("#225f85", "#93543b"), strict=True):
        label = "Marked" if row.marked else "Unmarked"
        axis.plot(
            REPETITIONS,
            row.exact,
            "-",
            color=color,
            alpha=0.55,
            label=f"{label}, exact",
        )
        axis.plot(
            REPETITIONS,
            [count / LOCAL_SHOTS for count in row.counts],
            "o",
            color=color,
            label=f"{label}, sampled",
        )
    axis.axhline(1 / 16, color="0.4", linestyle=":", label="1/16")
    axis.axvline(
        math.ceil(math.pi / 2 * math.sqrt(16 / 2)),
        color="0.4",
        linestyle="--",
        label="Source guide, k = 5",
    )
    axis.set(
        xlabel="Walk repetitions",
        ylabel="Probability of zero block",
        title="Marked-vertex walk on 16 vertices",
    )
    axis.set_xticks(REPETITIONS)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    figure.savefig(directory / "walk-zero-block.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("walk-block-encoding-figures")
    )
    arguments = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, arguments.figures)


if __name__ == "__main__":
    main()
