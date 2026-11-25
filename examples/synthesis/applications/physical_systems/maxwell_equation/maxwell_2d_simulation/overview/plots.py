"""Display the original three physical fields with the source normalization."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .workflow import Result


def fields(vector: torch.Tensor, title: str, destination: Path) -> None:
    phase = vector[vector.abs().argmax()]
    normalized = vector * (phase.abs() / phase) if float(phase.abs()) else vector
    components = normalized.reshape(4, 32, 32)[[0, 2, 3]].real
    maxima = components.abs().amax(dim=(1, 2))
    maxima = torch.where(maxima > 0, maxima, torch.ones_like(maxima))
    values = components / maxima[:, None, None]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    figure.suptitle(title)
    for axis, component, name in zip(axes, values, ("Ez", "Hx", "Hy"), strict=True):
        image = axis.imshow(
            component.numpy(), origin="lower", cmap="twilight", vmin=-1, vmax=1
        )
        axis.set_title(name)
        axis.set_xlabel("x grid index")
        axis.set_ylabel("y grid index")
        figure.colorbar(
            image, ax=axis, label="Field / its maximum absolute real amplitude"
        )
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def save(result: Result, directory: Path) -> tuple[str, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            "initial-fields.png",
            result.initial,
            "Original full-grid initial state at t=0",
        ),
        ("evolved-fields.png", result.projected, "Native Maxwell fields at t=5"),
        (
            "classical-fields.png",
            result.classical,
            "Independent classical Maxwell fields at t=5",
        ),
        (
            "quantum-fields.png",
            result.projected,
            "Quantum simulation on the original 32 by 32 grid",
        ),
    )
    for name, vector, title in figures:
        fields(vector, title, directory / name)
    return tuple(name for name, _, _ in figures)
