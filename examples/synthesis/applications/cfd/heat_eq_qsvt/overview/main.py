"""Execute, report and plot the original heat-equation workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .model import DTYPE, OBSERVED_STEPS, POINTS, TIME_STEPS, response
from .workflow import Result, run_workflow

if TYPE_CHECKING:
    from matplotlib.figure import Figure

OUTPUT = Path("artifacts/development/heat-equation-qsvt")


def report(result: Result) -> None:
    print("Grid points and original time samples:", POINTS, TIME_STEPS)
    print("Fresh Wx phases:", result.fit.phases)
    print("Fit maximum complex error:", result.fit.max_error)
    print("Rescaled block matrix norm error:", result.matrix_error)
    print("Projection norms:", result.projection_norms)
    print("First occupied basis indices:", result.gauge_indices)
    print("Displayed field rows:", len(result.fields))
    print("Last temperature row:", tuple(result.fields[-1].tolist()))
    print("Final native QSVT resources:", result.resources)


def save_figures(result: Result, output: Path = OUTPUT) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    model = result.model

    def save(figure: "Figure", name: str) -> None:
        path = output / f"{name}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        files.append(path)

    figure, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
    axis.plot(model.positions.tolist(), (model.initial * model.initial_norm).tolist())
    axis.set(title="Initial heat profile", xlabel="Position", ylabel="Temperature")
    save(figure, "initial")

    for label, color in (("left", "green"), ("right", "red")):
        figure, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
        axis.plot(
            model.times.tolist(), torch.zeros_like(model.times).tolist(), color=color
        )
        axis.set(
            title=f"Zero {label} boundary condition",
            xlabel="Time",
            ylabel="Temperature",
        )
        save(figure, f"{label}-boundary")

    figure, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
    axis.plot(model.positions.tolist(), model.initial.tolist())
    axis.set(title="Normalized input amplitudes", xlabel="Position", ylabel="Amplitude")
    save(figure, "normalized")

    for name, title, values in (
        ("operator", "Finite-difference operator", model.operator),
        ("projected", "Rescaled quantum block reconstruction", result.measured.real),
    ):
        figure, axis = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
        image = axis.imshow(values.numpy(), cmap="coolwarm")
        figure.colorbar(image, ax=axis)
        axis.set(title=title, xlabel="Input column", ylabel="Output row")
        save(figure, name)

    x = torch.linspace(-1, 1, 201, dtype=DTYPE)
    polynomial = result.fit.linear * x + result.fit.cubic * x**3
    measured = response(result.fit.phases, x).real
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    axis.plot(x.tolist(), polynomial.tolist(), label="Stabilized cubic")
    axis.plot(x.tolist(), measured.tolist(), "--", label="Fresh four-angle Wx response")
    axis.set(
        title="Phase fitting response", xlabel="Signal value", ylabel="Real response"
    )
    axis.legend()
    save(figure, "response")

    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for step in range(OBSERVED_STEPS):
        axis.plot(
            model.positions.tolist(), result.fields[step].tolist(), label=f"Step {step}"
        )
    axis.set(
        title="Signed field before each source iteration",
        xlabel="Position",
        ylabel="Signed projected field",
    )
    axis.legend()
    save(figure, "heat")
    return tuple(files)


def main() -> None:
    result = run_workflow()
    report(result)
    save_figures(result)


if __name__ == "__main__":
    main()
