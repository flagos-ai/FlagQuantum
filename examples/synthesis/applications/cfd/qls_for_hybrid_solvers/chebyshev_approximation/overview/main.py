"""Run four original-order CFD approximations and both native preparations."""

import argparse
from pathlib import Path

from .model import PREPARATION_ERROR
from .plots import save_figures
from .workflow import Result, run_workflow


def print_report(result: Result) -> None:
    nozzle = result.nozzle
    print("Matrix dimensions and entries:", 16, 16, 45)
    print("Banded normalization:", nozzle.alpha)
    print("Singular value interval:", nozzle.smallest, nozzle.largest)
    print("CKS power and theoretical degree:", nozzle.cks_power, nozzle.cks_degree)
    for item in result.approximations:
        print(
            f"{item.method} degree {item.degree} relative and uniform errors:",
            item.relative_error,
            item.uniform_error,
        )
    print("Preparation error bound:", PREPARATION_ERROR)
    print("Selector width and weights:", 7, result.fitted.numel())
    print(
        "Exact preparation RY and CX:",
        result.exact.resources.operation_counts["ry"],
        result.exact.resources.operation_counts["cx"],
    )
    print(
        "Approximate preparation RY and CX:",
        result.approximate.resources.operation_counts["ry"],
        result.approximate.resources.operation_counts["cx"],
    )
    print("Uniform low bits:", result.approximate.bits_grouped)
    print(
        "Measured amplitude and probability L2 errors:",
        result.approximate.amplitude_error,
        result.approximate.probability_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("chebyshev-figures"))
    args = parser.parse_args()
    result = run_workflow()
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
