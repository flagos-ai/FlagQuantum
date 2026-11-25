"""Plot the observed costs and both source lattice paths."""

from pathlib import Path
from typing import Any

from .model import RESIDUES, lattice
from .workflow import Result


def save_figures(result: Result, output: Path) -> tuple[Path, ...]:
    """Write the optimization trace, matched cost histograms, and folds."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(figure: Any, name: str) -> None:
        path = output / f"{name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(1, len(result.evaluations) + 1),
        [row.lower_tail_mean for row in result.evaluations],
    )
    ax.set(
        xlabel="Actual objective query",
        ylabel="Lower 70% mean cost",
        title="Folding optimization",
    )
    save(fig, "trace")

    fig, ax = plt.subplots(figsize=(7, 4))
    for draws, label in ((result.optimized, "Optimized"), (result.uniform, "Uniform")):
        ax.hist(
            [result.energies[word] for word in draws.tolist()],
            bins=50,
            range=(min(result.energies), max(result.energies)),
            weights=[1 / draws.numel()] * draws.numel(),
            alpha=0.6,
            label=label,
        )
    ax.set(
        xlabel="Folding cost",
        ylabel="Sampled probability",
        title="Sampled folding costs",
    )
    ax.legend()
    save(fig, "costs")

    fig = plt.figure(figsize=(8, 4))
    panels = (
        (result.observed_word, "Best observed fold"),
        (result.exact_words[0], "Exact minimum"),
    )
    for index, (word, title) in enumerate(panels, start=1):
        axes = fig.add_subplot(1, 2, index, projection="3d")
        x, y, z = zip(*lattice(word), strict=True)
        axes.plot(x, y, z, marker="o")
        for label, point in zip(RESIDUES, lattice(word), strict=True):
            axes.text(*point, label)
        axes.set_title(f"{title}\nCost {result.energies[word]}", pad=16)
        axes.set_axis_off()
    save(fig, "folds")
    return tuple(paths)
