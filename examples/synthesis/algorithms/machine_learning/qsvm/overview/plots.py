"""Visualize the original data splits and the native predictions."""

from pathlib import Path

import matplotlib.pyplot as plt

from .datasets import Dataset, Split
from .workflow import Score

COLORS = ("#2864aa", "#c45b24")


def synthetic(data: Split, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
    try:
        for index, (label, group) in enumerate(
            (
                ("Training", data.train),
                ("Test", data.test),
                ("Prediction inputs", data.predict),
            )
        ):
            axis = axes[index]
            if index == 2:
                axis.scatter(
                    group.points[:, 0],
                    group.points[:, 1],
                    color="#494d54",
                    s=15,
                    alpha=0.8,
                )
            else:
                for category, color in enumerate(COLORS):
                    points = group.points[group.labels == category]
                    axis.scatter(
                        points[:, 0],
                        points[:, 1],
                        color=color,
                        s=15,
                        label=str(category),
                    )
            axis.set_title(label)
            axis.set_xlabel("Feature 1")
            axis.set_xlim(-1.03, -0.34)
            axis.set_ylim(-1.03, 0.03)
        axes[0].set_ylabel("Feature 2")
        axes[0].legend(title="Class")
        figure.tight_layout()
        figure.savefig(directory / "synthetic-data.png", dpi=130)
    finally:
        plt.close(figure)


def prediction(data: Dataset, result: Score, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    try:
        for category, color in enumerate(COLORS):
            points = data.points[result.predictions == category]
            axis.scatter(points[:, 0], points[:, 1], color=color, label=str(category))
        axis.set_title("Synthetic predictions by the Bloch kernel")
        axis.set_xlabel("Feature 1")
        axis.set_ylabel("Feature 2")
        axis.legend(title="Predicted class")
        figure.tight_layout()
        figure.savefig(directory / "synthetic-prediction.png", dpi=130)
    finally:
        plt.close(figure)


def iris(data: Split, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    try:
        for column, (group, split) in enumerate(
            ((data.train, "train"), (data.test, "test"))
        ):
            for category, (name, color, marker) in enumerate(
                zip(("Versicolor", "Virginica"), COLORS, ("s", "o"), strict=True)
            ):
                points = group.points[group.labels == category]
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    marker=marker,
                    facecolors=color if column else "none",
                    edgecolors=color,
                    label=f"{name} {split}",
                    s=30,
                )
        axis.set_title("Iris inputs for the two-feature classifier")
        axis.set_xlabel("Scaled petal length (rad)")
        axis.set_ylabel("Scaled petal width (rad)")
        axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
        figure.tight_layout()
        figure.savefig(directory / "iris-data.png", dpi=130)
    finally:
        plt.close(figure)
