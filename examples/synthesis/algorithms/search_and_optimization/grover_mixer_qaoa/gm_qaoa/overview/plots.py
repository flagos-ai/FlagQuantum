"""Plot actual finite objectives and sampled distributions for both ansatzes."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import SHOTS, Run


def objective_trace(result: Run, label: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row.cost for row in result.trace], marker=".", linewidth=1)
    ax.set(title=label, xlabel="Actual objective query", ylabel="Sampled mean cost")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def distributions(ordinary: Run, grover: Run, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    words = sorted(
        {row.word for result in (ordinary, grover) for row in result.observations}
    )
    for result, sign, label in (
        (ordinary, -1, "Ordinary QAOA"),
        (grover, 1, "Grover mixer"),
    ):
        frequencies = {row.word: row.count / SHOTS for row in result.observations}
        ax.bar(
            [i + 0.19 * sign for i in range(len(words))],
            [frequencies.get(word, 0.0) for word in words],
            width=0.37,
            label=label,
        )
    ax.set(
        title="Observed samples from both optimized circuits",
        xlabel="Four-bit word (MSB first)",
        ylabel="Observed probability",
        xticks=range(len(words)),
        xticklabels=[f"{word:04b}" for word in words],
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
