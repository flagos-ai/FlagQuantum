"""Plot actual gradient readout and all original parameter sweeps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .workflow import Model, Observation, PhaseRow


def save(figure: Figure, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    path = directory / (name + ".png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def classical_axes(axis: Axes, model: Model) -> None:
    indices = [-8 + 16 * i / 999 for i in range(1000)]
    axis.plot(
        indices,
        [model.normalized(x) - model.normalized(0) for x in indices],
        color="lightgray",
        label="Original function",
    )
    axis.set_xticks(range(-4, 4))
    axis.set_xlabel("x (index)")
    axis.set_ylabel("f (normalized)")
    top = axis.secondary_xaxis(
        "top",
        functions=(
            lambda x: np.asarray(x, dtype=np.float64) * model.window / 8,
            lambda x: np.asarray(x, dtype=np.float64) * 8 / model.window,
        ),
    )
    top.set_xlabel("x (real)")
    right = axis.secondary_yaxis(
        "right",
        functions=(
            lambda y: (np.asarray(y, dtype=np.float64) + model.normalized(0))
            * model.m
            * model.window
            / 8,
            lambda y: np.asarray(y, dtype=np.float64) * 8 / (model.m * model.window)
            - model.normalized(0),
        ),
    )
    right.set_ylabel("f (real, unnormalized)")


def phase_plot(
    model: Model, rows: tuple[PhaseRow, ...], directory: Path, name: str
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    classical_axes(axis, model)
    axis.plot(
        [r.x for r in rows],
        [r.theoretical for r in rows],
        "o",
        label="Theoretical samples",
    )
    axis.plot(
        [r.x for r in rows],
        [r.unwrapped_phase for r in rows],
        ".",
        label="Measured unwrapped phase",
    )
    axis.set_xlim(-8, 8)
    axis.set_ylim(-4, 4)
    axis.legend()
    return save(figure, directory, name)


def gradient_plot(
    model: Model, observations: tuple[Observation, ...], directory: Path, name: str
) -> Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    classical_axes(left, model)
    indices = list(range(-4, 4))
    left.plot(
        indices,
        [model.normalized(x) - model.normalized(0) for x in indices],
        "o",
        label="Theoretical samples",
    )
    left.set_xlim(-8, 8)
    left.set_ylim(-4, 4)
    left.legend()
    left.set_title("Theoretical phases")
    right.bar(
        [r.values[0] for r in observations],
        [100 * r.frequency for r in observations],
        color="lightblue",
        label="Actual measurement counts",
    )
    right.axvline(
        model.linear * 8 / model.m,
        color="green",
        linestyle="dashed",
        label="Analytical gradient",
    )
    right.set_xlabel("x (index)")
    right.set_ylabel("Percentage of shots (%)")
    right.set_ylim(0, 100)
    right.set_xlim(-5, 4)
    right.set_title("Measurement histogram")
    right.legend()
    return save(figure, directory, name)


def sweep_plot(
    values: tuple[float, ...], rates: tuple[float, ...], parameter: str, directory: Path
) -> Path:
    if parameter not in ("m", "l") or len(values) != len(rates):
        raise ValueError("use matching source m/l sweep results")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(values, rates, "o-", label="Actual sampled success")
    low, high = (0.62, 3.2) if parameter == "m" else (0.0, 0.28)
    axis.axvspan(
        low, high, color="green", alpha=0.15, label="Source theoretical interval"
    )
    axis.set_xlabel(parameter + " parameter")
    axis.set_ylabel("Success rate")
    axis.set_title("Success rate vs " + parameter)
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.3)
    axis.legend()
    return save(figure, directory, "sweep_" + parameter)
