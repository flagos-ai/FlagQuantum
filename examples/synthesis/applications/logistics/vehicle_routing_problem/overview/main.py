"""Execute the full original vehicle-routing workflow."""

import argparse
from pathlib import Path

from . import plots, problem, workflow


def report(result: workflow.Result) -> None:
    data = result.problem
    print(f"Locations: {data.positions}")
    print(f"Distance normalization: {data.scale:.15g}")
    print("Six-by-six normalized distance matrix:")
    for row in data.distances:
        print(" ".join(f"{value:.9f}" for value in row))
    print(
        f"Optimizer queries: {len(result.trace)}, budget reached={result.budget_reached}, converged={result.converged}"
    )
    print(f"Actual draws including final sample: {result.sampled_shots}")
    print(
        f"Lowest sampled objective: {min(row.sampled_cost for row in result.trace):.12g}"
    )
    print(f"Final parameters: {result.parameters}")
    print("Twenty lowest-cost observed assignments:")
    for word, count in result.counts[:20]:
        print(
            f"word={word:016b}, count={count}, cost={problem.scalar_cost(data,word):.12g}, valid={problem.valid(word)}"
        )
    print(f"Selected actual word: {result.selected_word:016b}")
    for vehicle in range(2):
        print(f"Vehicle {vehicle} route: {problem.route(result.selected_word,vehicle)}")
    best, optimal = problem.classical_reference(data)
    print(
        f"Independent feasible cost: {best:.12g}, exact minimizer count={len(optimal)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("routing-figures"))
    args = parser.parse_args()
    result = workflow.optimize(problem.fresh())
    report(result)
    plots.save(result, args.figures)


if __name__ == "__main__":
    main()
