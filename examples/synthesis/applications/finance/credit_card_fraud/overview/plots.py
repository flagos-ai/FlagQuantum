"""Display both complete t-SNE embeddings using their actual labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .model import LABELS

COLORS = ("#376ea7", "#d36b34")


def scatter(
    embedding: NDArray[np.float64],
    labels: NDArray[np.int64],
    title: str,
    target: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(5.8, 5.0))
    try:
        for category, (name, color) in enumerate(zip(LABELS, COLORS, strict=True)):
            group = embedding[labels == category]
            axes.scatter(
                group[:, 0],
                group[:, 1],
                s=23,
                color=color,
                label=f"{name} ({len(group)})",
            )
        axes.set_title(title)
        axes.set_xlabel("t-SNE coordinate 1")
        axes.set_ylabel("t-SNE coordinate 2")
        axes.axis("equal")
        axes.legend()
        figure.tight_layout()
        figure.savefig(target, dpi=140)
    finally:
        plt.close(figure)
