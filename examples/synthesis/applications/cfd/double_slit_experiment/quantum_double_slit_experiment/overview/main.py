"""Run the original-resolution double-slit inverse and animate the resulting wave."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import hsv_to_rgb
from matplotlib.image import AxesImage

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)

from .model import DEGREE, GRID, REAL
from .workflow import Result, run

FRAMES = 60
FPS = 30


def normalized_field(wave: torch.Tensor) -> torch.Tensor:
    highest = wave[wave.abs().argmax()]
    return (wave * highest.conj() / highest.abs()).reshape(GRID, GRID)


def color_field(wave: torch.Tensor) -> torch.Tensor:
    phase = normalized_field(wave)
    scaled = (phase.abs() / torch.quantile(phase.abs(), 0.5)).tanh()
    hue = (torch.angle(phase) / (2 * math.pi)).remainder(1)
    hsv = torch.stack((hue, torch.full_like(hue, 0.85), scaled / scaled.max()), dim=-1)
    return torch.as_tensor(hsv_to_rgb(hsv.numpy()), dtype=REAL)


def print_report(result: Result) -> None:
    print("Grid points and QSVT degree:", GRID**2, DEGREE)
    print("Phase count:", result.inverse.phases.phases.numel())
    print(
        "Angular frequency, matrix diagonal and normalization:",
        result.model.omega,
        result.model.diagonal,
        result.model.alpha,
    )
    print(
        "Minimum and maximum singular values:",
        result.inverse.minimum_sigma,
        result.inverse.maximum_sigma,
    )
    print("Phase response residual:", result.inverse.phases.residual)
    print(
        "SVD and native fidelities:", result.inverse.expected_fidelity, result.fidelity
    )
    print("Postselection probability:", result.selected_probability)
    print(
        "Full and spectral absolute errors:", result.full_error, result.spectral_error
    )
    print("Workspace probability:", result.work_probability)
    print(
        "Simulator resources:",
        result.resources.n_qubits,
        result.resources.n_operations,
        result.resources.depth,
        result.resources.max_operation_width,
    )
    print("Animation frames and nominal FPS:", FRAMES, FPS)


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.2), layout="constrained")
    x = torch.linspace(0, result.inverse.maximum_sigma, 2000, dtype=REAL)
    valid = x[x >= result.inverse.minimum_sigma]
    axis.plot(
        x.tolist(),
        response(x, result.inverse.phases.phases).tolist(),
        label="Fresh degree-191 response",
    )
    target = 0.75 * result.inverse.minimum_sigma / valid
    axis.plot(valid.tolist(), target.tolist(), "--", label="Scaled reciprocal")
    axis.axvline(
        result.inverse.minimum_sigma,
        linestyle=":",
        color="0.35",
        label="Minimum singular value",
    )
    axis.set(
        xlabel="Singular value",
        ylabel="Encoded response",
        title="Inverse polynomial for the 16 × 16 wave grid",
    )
    axis.legend()
    figure.savefig(directory / "inverse-poly.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), layout="constrained")
    for panel, wave, label in zip(
        axes,
        (result.model.reference, result.wave),
        ("Independent physical solve", "FlagQuantum postselection"),
        strict=True,
    ):
        panel.imshow(color_field(wave), origin="lower", interpolation="nearest")
        panel.set(xlabel="Grid y", ylabel="Grid x", title=label)
    figure.suptitle("Double-slit wave field")
    figure.savefig(directory / "wave-fields.png", dpi=140)
    plt.close(figure)


def save_animation(result: Result, directory: Path) -> None:
    wave = normalized_field(result.wave)
    level = torch.quantile(wave.abs(), 0.7)
    figure, axis = plt.subplots(figsize=(4.8, 4.8), layout="constrained")
    image = axis.imshow(
        (wave.real / level).tanh(),
        origin="lower",
        aspect="equal",
        cmap="twilight",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set(
        xlabel="Grid y", ylabel="Grid x", title="Real part of the postselected wave"
    )

    def update(frame: int) -> tuple[AxesImage]:
        angle = -2 * math.pi * frame / FRAMES
        rotation = complex(math.cos(angle), math.sin(angle))
        image.set_data(((wave * rotation).real / level).tanh())
        return (image,)

    animation = FuncAnimation(
        figure, update, frames=range(FRAMES), interval=1000 / FPS, blit=False
    )
    animation.save(directory / "wave-cycle.gif", writer=PillowWriter(fps=FPS), dpi=95)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("double-slit-figures"))
    options = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, options.figures)
    save_animation(result, options.figures)


if __name__ == "__main__":
    main()
