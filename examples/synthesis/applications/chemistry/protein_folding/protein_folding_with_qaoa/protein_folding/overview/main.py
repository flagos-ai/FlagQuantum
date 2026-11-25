"""Run the complete six-residue native folding workflow."""

from pathlib import Path

from .model import directions, lattice
from .plots import save_figures
from .workflow import SHOTS, Result, run_workflow

OUTPUT = Path("artifacts/development/protein-folding-qaoa")


def report(result: Result) -> None:
    """Print actual optimization, sampling, and classical reference values."""
    print(
        "Objective queries and draws:",
        len(result.evaluations),
        len(result.evaluations) * SHOTS,
    )
    print(
        "Optimized and uniform draws:", result.optimized.numel(), result.uniform.numel()
    )
    print("Budget stop and convergence:", result.budget_reached, result.converged)
    print(
        "Observed word and cost:",
        result.observed_word,
        result.energies[result.observed_word],
    )
    print("Observed directions:", directions(result.observed_word))
    print("Observed coordinates:", lattice(result.observed_word))
    print(
        "Exact words and cost:",
        result.exact_words,
        result.energies[result.exact_words[0]],
    )


def main() -> None:
    result = run_workflow()
    report(result)
    for path in save_figures(result, OUTPUT):
        print(path)


if __name__ == "__main__":
    main()
