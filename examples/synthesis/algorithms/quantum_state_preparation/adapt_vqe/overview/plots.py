"""Plot each measured adaptive energy or gradient trajectory."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def save_trace(
    values: Sequence[float],
    reference: float,
    *,
    title: str,
    quantity: str,
    reference_label: str,
    path: Path,
) -> Path:
    """Show the actual measurements and the applicable reference at each step."""
    figure, axis = plt.subplots(figsize=(7.8, 4.2))
    indices = list(range(1, len(values) + 1))
    axis.plot(indices, values, "o-", label=quantity)
    axis.axhline(reference, color="#686868", linestyle="--", label=reference_label)
    axis.set(title=title, xlabel="Adaptive step", ylabel=quantity, xticks=indices)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
