"""Run both original matrix-HHL routes, absolute checks and solution plots."""

from pathlib import Path

from .ansatz import Route, build_program
from .physics import MATRIX, PAULI_TERMS, REPETITIONS, B
from .workflow import extract_solution, plot_solution, verify_program


def run_example() -> None:
    print("Matrix:", MATRIX, "Normalized RHS:", B)
    print("Pauli expansion:", PAULI_TERMS, "Power-dependent repetitions:", REPETITIONS)
    routes: tuple[Route, ...] = ("exact", "trotter")
    for route in routes:
        program = build_program(route)
        print(route, verify_program(program, route))
        solution = extract_solution(program, check=route != "exact")
        print("Actual solution:", solution)
        print(
            plot_solution(solution, route, Path("artifacts/development/algorithm_hhl"))
        )


if __name__ == "__main__":
    run_example()
