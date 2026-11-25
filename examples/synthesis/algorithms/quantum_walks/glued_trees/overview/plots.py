"""Draw the two original exit probability traces from finite samples."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Point, exit_word


def plot_walk(qubits: int, rows: tuple[Point, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.plot(
        [row.time for row in rows],
        [row.sampled_probability for row in rows],
        "o-",
        label="Observed shots",
    )
    axes.plot(
        [row.time for row in rows],
        [row.exact_probability for row in rows],
        "--",
        label="Exact state probability",
    )
    axes.set(
        xlabel="Evolution time",
        ylabel=f"Proportion for exit word {exit_word(qubits)}",
        title=f"Glued trees on {qubits} qubits",
    )
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
