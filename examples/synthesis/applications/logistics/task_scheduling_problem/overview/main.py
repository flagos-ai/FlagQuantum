"""Execute both original task-scheduling instances with native qfunc."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(results: tuple[workflow.Result, workflow.Result]) -> None:
    for result in results:
        model = result.problem
        print(
            f"{model.name}: {model.tasks} tasks, {len(model.capacity)} slots, {model.width} quantum bits, eight layers"
        )
        print(
            f"Capacity={model.capacity}, workloads={model.workload}, active variables={model.active}"
        )
        print(
            f"Actual queries={len(result.trace)}, CVaR alpha={result.quantile}, budget reached={result.stopped}, converged={result.converged}"
        )
        print(
            f"Actual draws={(len(result.trace)+1)*workflow.SHOTS}, final sample={sum(count for _,count in result.counts)}"
        )
        print(
            f"Observed selected cost={result.selected_cost}, feasible={result.selected_valid}, matrix={result.selected_matrix}"
        )
        print(
            f"Independent classical finish={result.classical_cost}, matrix={result.classical_matrix}"
        )


def write(results: tuple[workflow.Result, workflow.Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def run() -> tuple[workflow.Result, workflow.Result]:
    return workflow.optimize(False), workflow.optimize(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("scheduling-figures"))
    parser.add_argument("--results", type=Path, default=Path("scheduling-results.json"))
    args = parser.parse_args()
    results = run()
    report(results)
    plots.save(results, args.figures)
    write(results, args.results)


if __name__ == "__main__":
    main()
