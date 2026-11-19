"""Present the original system matrices, vector amplitudes, and population curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .model import DT, N


def save_matrix_views(
    matrices: tuple[tuple[str, torch.Tensor], ...], path: Path
) -> Path:
    """Pair a matrix and its input vector where they share a model stage."""
    figure, axes = plt.subplots(
        1, len(matrices), figsize=(4.7 * len(matrices), 3.8), squeeze=False
    )
    for axis, (title, matrix) in zip(axes[0], matrices, strict=True):
        image = axis.imshow(matrix.real.numpy(), aspect="auto", cmap="coolwarm")
        axis.set(title=title, ylabel="Row")
        if matrix.shape[1] == 1:
            axis.set_xticks([])
        else:
            axis.set_xlabel("Column")
            stride = max(1, matrix.shape[1] // 8)
            axis.set_xticks(list(range(0, matrix.shape[1], stride)))
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def save_vectors(series: tuple[tuple[str, torch.Tensor], ...], path: Path) -> Path:
    """Plot the signed components on all source basis words."""
    figure, axis = plt.subplots(figsize=(8, 4))
    for label, vector in series:
        axis.plot(range(vector.numel()), vector.real.tolist(), marker=".", label=label)
    axis.set(
        xlabel="Basis word",
        ylabel="Solution component",
        title="Lanchester solution vector",
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def save_populations(
    classical: torch.Tensor, path: Path, quantum: torch.Tensor | None = None
) -> Path:
    """Use the eight original times and the two source population groups."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    years = [DT * index for index in range(N)]
    for label, group, color in (
        ("Rabbits", slice(0, N), "tab:blue"),
        ("Foxes", slice(N, 2 * N), "tab:orange"),
    ):
        axis.plot(
            years, classical[group].real.tolist(), label=f"{label} direct", color=color
        )
        if quantum is not None:
            axis.plot(
                years,
                quantum[group].real.tolist(),
                "o--",
                label=f"{label} HHL",
                color=color,
            )
    axis.set(
        title="Predator and prey populations",
        xlabel="Time (years)",
        ylabel="Population (thousands)",
    )
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
