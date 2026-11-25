"""Run original estimation and sequential approximate/exact-QPE overlap experiment."""

from pathlib import Path

from .ansatz import build_program, repetition_table, source_eigenpair
from .plots import plot_phase
from .workflow import verify_and_sample


def run_example() -> None:
    print("Fresh source eigendecomposition:", source_eigenpair())
    print(
        "Power repetitions:",
        tuple((p, repetition_table()[p]) for p in (1, 2, 4, 8, 16)),
    )
    estimate = verify_and_sample(build_program(), seed=4100)
    print(estimate)
    plot_phase(estimate, Path("artifacts/development/flexible_qpe"))
    overlap = verify_and_sample(build_program(True), comparison=True, seed=4101)
    print(overlap)
    print(
        "Dropped data and phase registers remain private environment; no clean-work claim"
    )


if __name__ == "__main__":
    run_example()
