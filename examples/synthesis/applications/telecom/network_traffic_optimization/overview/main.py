"""Run every original traffic-network workflow stage locally."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, problem, workflow


def report(result: workflow.Result) -> None:
    print(
        f"Network: {len(problem.NODES)} nodes, {len(problem.EDGES)} edges, {len(problem.DEMANDS)} demands"
    )
    print(f"Integral-flow reference word: {result.classical_word:024b}")
    print(
        f"Source MILP objective and actual latency: {result.classical_objective}, {result.classical_latency}"
    )
    print(
        f"Optimizer queries: {len(result.trace)}, budget reached={result.stopped}, converged={result.converged}"
    )
    print(f"Actual quantum draws: {(len(result.trace)+1)*problem.SHOTS}")
    print(f"Final parameters: {result.parameters}")
    print(f"Final sample counts: {sum(count for _,count in result.counts)}")
    print("Fifteen most frequent observed outcomes:")
    for word, count in result.counts[:15]:
        print(
            f"word={word:024b}, count={count}, valid={problem.valid(word)}, cost={problem.cost(word):.12g}"
        )
    print(f"Observed feasible assignments: {len(result.feasible)}")
    print(
        f"Observed feasible probability: {sum(count for _,count in result.feasible)/problem.SHOTS:.8f}"
    )
    for word, count in result.feasible[:3]:
        print(f"Observed feasible word={word:024b}, count={count}")
        for demand in range(3):
            print(f"Demand {demand+1}: {problem.chosen_edges(word,demand)}")


def write(result: workflow.Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("traffic-figures"))
    parser.add_argument("--results", type=Path, default=Path("traffic-results.json"))
    args = parser.parse_args()
    result = workflow.optimize()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
