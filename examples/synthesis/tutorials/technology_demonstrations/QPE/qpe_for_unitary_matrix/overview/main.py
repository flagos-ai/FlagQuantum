"""Run all eight original precision points and save the qualified cost plot."""

from pathlib import Path

from .ansatz import build_program
from .physics import EIGENVALUES, PRECISIONS
from .workflow import plot_study, verify_program


def run_example() -> None:
    print("Original unshifted spectrum:", EIGENVALUES)
    points = tuple(
        verify_program(build_program(precision), precision) for precision in PRECISIONS
    )
    for point in points:
        print(point)
    print(plot_study(points, Path("artifacts/development/matrix_qpe_precision")))


if __name__ == "__main__":
    run_example()
