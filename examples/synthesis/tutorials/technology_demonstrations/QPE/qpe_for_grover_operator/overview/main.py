"""Run the original Grover QPE precision and synthesis-objective study."""

from pathlib import Path
from typing import Literal

from .ansatz import build_program
from .workflow import plot_study, verify_program


def run_example() -> None:
    objectives: tuple[Literal["width", "depth"], ...] = ("width", "depth")
    points = tuple(
        verify_program(build_program(precision, objective), precision, objective)
        for precision in range(1, 6)
        for objective in objectives
    )
    for point in points:
        print(point)
    print(plot_study(points, Path("artifacts/development/grover_qpe")))


if __name__ == "__main__":
    run_example()
