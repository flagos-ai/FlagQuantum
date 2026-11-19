"""Plot the original topology and actual classical and sampled routes."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from . import problem, workflow

COLORS = ("tab:red", "tab:cyan", "teal")


def draw(directory: Path, label: str, title: str, word: int | None) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for edge, (first, second) in enumerate(problem.EDGES):
        owner = (
            None
            if word is None
            else next((d for d in range(3) if word >> (8 * d + edge) & 1), None)
        )
        color = "0.7" if owner is None else COLORS[owner]
        x, y = problem.POSITIONS[first]
        xx, yy = problem.POSITIONS[second]
        arrow = FancyArrowPatch(
            (x, y),
            (xx, yy),
            arrowstyle="-|>",
            mutation_scale=18,
            shrinkA=21,
            shrinkB=21,
            linewidth=1 if owner is None else 3,
            color=color,
        )
        ax.add_patch(arrow)
        ax.text(
            (x + xx) / 2,
            (y + yy) / 2 + 0.04,
            str(problem.LATENCY[edge]),
            ha="center",
            fontsize=11,
        )
    for name, (x, y) in problem.POSITIONS.items():
        ax.scatter(x, y, s=850, color="#D7F75B", zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=13, zorder=4)
    for index, (source, sink) in enumerate(problem.DEMANDS):
        ax.plot(
            [], [], color=COLORS[index], label=f"Demand {index+1}: {source} → {sink}"
        )
    ax.set(xlim=(-0.3, 3.3), ylim=(-0.15, 1.4), title=title)
    ax.axis("off")
    if word is not None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    filename = label + ".png"
    fig.savefig(directory / filename, dpi=140)
    plt.close(fig)
    return filename


def save(result: workflow.Result, directory: Path) -> tuple[str, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    files = [
        draw(directory, "topology", "Six-node, three-demand traffic network", None),
        draw(
            directory,
            "classical",
            "Independent integral-flow solution",
            result.classical_word,
        ),
    ]
    for i, (word, count) in enumerate(result.feasible[:3]):
        files.append(
            draw(
                directory,
                f"quantum-{i+1}",
                f"Observed feasible routing • {count} / {problem.SHOTS} shots",
                word,
            )
        )
    return tuple(files)
