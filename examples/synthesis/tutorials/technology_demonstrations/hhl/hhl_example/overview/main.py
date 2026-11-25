"""Run every original HHL precision with independent states and actual costs."""

from pathlib import Path

from .ansatz import build_program
from .physics import PRECISIONS, RAW_B_NORM
from .workflow import plot_study, verify_program


def run_example() -> None:
    print("Rounded source RHS norm:", RAW_B_NORM)
    print("Preparation explicitly normalizes the original rounded vector")
    points = tuple(
        verify_program(build_program(precision), precision) for precision in PRECISIONS
    )
    for point in points:
        print(point)
    print(plot_study(points, Path("artifacts/development/hhl_precision")))


if __name__ == "__main__":
    run_example()
