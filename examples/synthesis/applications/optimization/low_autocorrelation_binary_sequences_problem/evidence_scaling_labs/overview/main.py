"""Generate the complete LABS model, compare CX candidates and run QAOA."""

import argparse
from pathlib import Path

from . import checks, model, plots, workflow


def report(result: workflow.Result, errors: tuple[float, ...]) -> None:
    print("Original size and layers:", (model.WIDTH, model.LAYERS))
    print("Generated Hamiltonian terms:", len(model.cost_terms()))
    print(
        "Actual CX candidates:",
        tuple((item.schedule.value, item.cx) for item in result.candidates),
    )
    print("Selected schedule:", result.selected.schedule.value)
    print(
        "Selected width, total gates and depth:",
        (
            result.selected.resources.n_qubits,
            result.selected.resources.n_operations,
            result.selected.resources.depth,
        ),
    )
    print("Complete candidate amplitude errors:", errors)
    print(
        "Objective queries and finite final draws:",
        (len(result.trace), sum(row.count for row in result.samples)),
    )
    print("Budget stop and convergence:", (result.budget_stop, result.converged))
    print("Best evaluated mean cost:", result.chosen.cost)
    print(
        "Best observed word, cost and multiplicity:",
        (result.samples[0].word, result.samples[0].cost, result.samples[0].count),
    )
    print(
        "Separate exhaustive minimum:", (result.exact_minimum, len(result.exact_words))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("labs-results"))
    args = parser.parse_args()
    result = workflow.run()
    report(result, checks.verify(result.candidates))
    print("First five observed rows:", result.samples[:5])
    plots.save(result, args.figures)


if __name__ == "__main__":
    main()
