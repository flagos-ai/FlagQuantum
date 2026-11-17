"""Run the explicit and precision-driven QSVT matrix inverses."""

import argparse
from pathlib import Path

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    evaluate,
)

from .model import Model
from .workflow import SCALE, Inverse, inverse_coefficients, run_workflow, source_rhs


def print_report(model: Model, manual: Inverse, automatic: Inverse) -> None:
    print(
        "Bare and effective condition numbers:",
        model.condition,
        model.effective_condition,
    )
    print("Data bits, selector bits and block scale:", 2, 4, model.encoding.alpha)
    for name, result in (("Explicit", manual), ("Precision-driven", automatic)):
        print(
            f"{name} degree and phase count:",
            result.degree,
            result.phase_fit.phases.numel(),
        )
        print(f"{name} response max error:", result.response_error)
        print(
            f"{name} unscaled inverse max error:",
            result.response_error * model.condition / SCALE,
        )
        print(f"{name} dense response fit max error:", result.phase_fit.residual)
        print(
            f"{name} full and spectral max errors:",
            result.full_state_error,
            result.spectral_error,
        )
        print(f"{name} solution L2 error:", result.solution_error)
        print(f"{name} projection probability:", result.projection_probability)
        print(f"{name} solution:", tuple(map(float, result.solution)))


def save_response(model: Model, manual: Inverse, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    origin, high = (
        1 / model.effective_condition,
        float(model.spectrum[0]) / model.encoding.alpha,
    )
    grid = torch.linspace(origin, high, 512, dtype=torch.float64)
    exact = SCALE * origin / grid
    polynomial = evaluate(inverse_coefficients(model, manual.degree), grid)
    figure, (main, residual) = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    main.plot(grid, exact, color="black", label="Scaled reciprocal")
    main.plot(grid, polynomial, "--", label=f"Degree {manual.degree}")
    main.set(ylabel="Response", title="QSVT inverse response")
    main.legend()
    residual.plot(grid, polynomial - exact)
    residual.set(xlabel="Normalized singular value", ylabel="Approximation error")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_solution(
    model: Model, manual: Inverse, automatic: Inverse, path: Path
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    expect = torch.linalg.solve(model.matrix, source_rhs().to(model.matrix.dtype)).real
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    index = range(expect.numel())
    axes.plot(index, expect, "o", label="Exact dense solve")
    axes.plot(index, manual.solution, "x", label=f"Explicit degree {manual.degree}")
    axes.plot(
        index,
        automatic.solution,
        "+",
        label=f"Precision-driven degree {automatic.degree}",
    )
    axes.set(
        xticks=list(index),
        xlabel="Component",
        ylabel="Solution",
        title="Four-dimensional matrix inverse",
    )
    axes.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("matrix-inversion-figures")
    )
    args = parser.parse_args()
    model, manual, automatic = run_workflow()
    print_report(model, manual, automatic)
    save_response(model, manual, args.figures / "inverse-response.png")
    save_solution(model, manual, automatic, args.figures / "matrix-solution.png")


if __name__ == "__main__":
    main()
