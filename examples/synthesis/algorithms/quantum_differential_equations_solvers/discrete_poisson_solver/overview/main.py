"""Execute the full original-sized Poisson workflow and honest diagnostics."""

from pathlib import Path

from .ansatz import build_program
from .workflow import extract_solution, plot_solution, plot_source, verify_program


def run_example() -> None:
    directory = Path("artifacts/development/discrete_poisson")
    program = build_program()
    print(verify_program(program))
    solution = extract_solution(program)
    print(solution)
    print(plot_source(directory), plot_solution(solution, directory))
    print("Source-scale relative distance:", solution.relative_distance)


if __name__ == "__main__":
    run_example()
