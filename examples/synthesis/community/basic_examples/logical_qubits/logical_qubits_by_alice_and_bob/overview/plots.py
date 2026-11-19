"""Display all105 actually sampled source parameter points."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import workflow
from .noise import Parameters


def save(result: workflow.Result, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    color = LinearSegmentedColormap.from_list(
        "cat_logical_error", [(0, 0.5, 0), (0.9, 0.5, 0), (0.9, 0, 0)], N=100
    )
    points = {point.parameters: point for point in result.grid}
    for photons in workflow.PHOTONS:
        values = np.array(
            [
                [
                    points[Parameters(distance, photons, kappa)].observed_error
                    for kappa in workflow.KAPPA_2
                ]
                for distance in workflow.DISTANCES
            ]
        )
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        image = axis.imshow(
            values, origin="lower", aspect="auto", cmap=color, vmin=0, vmax=0.5
        )
        axis.set_xticks(range(3), labels=["10³", "10⁴", "10⁵"])
        axis.set_yticks(range(7), labels=[str(value) for value in workflow.DISTANCES])
        axis.set_xlabel("Two-photon / one-photon loss ratio, kappa2 / kappa1")
        axis.set_ylabel("Repetition-code distance")
        axis.set_title(f"Logical-cat SWAP test: mean photons = {photons}")
        for row in range(7):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        figure.colorbar(
            image, ax=axis, label="Observed error rate, 10,000 actual draws / point"
        )
        figure.savefig(destination / f"photons-{photons}.png", dpi=140)
        plt.close(figure)
