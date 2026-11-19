"""Execute the original 24-bit network resiliency sampling workflow."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print("Original network: five nodes, seven edges, two demands, 24 bits, 20 layers")
    print(
        f"Actual objective queries={len(result.trace)}, stopped={result.stopped}, converged={result.converged}"
    )
    print(
        f"Actual draws={len(result.trace) * result.objective_shots + result.final_shots}, final draws={sum(count for _, count in result.counts)}"
    )
    print(f"Observed objective trace={tuple(value for _, value in result.trace)}")
    print(
        f"Valid observed outcomes={len(result.feasible)}, valid probability={sum(count for _, count in result.feasible) / result.final_shots}"
    )
    for rank, (word, count) in enumerate(result.counts[:100]):
        if (word, count) in result.feasible:
            print(
                f"Valid sample rank={rank}, word={word}, counts={count}, probability={count / result.final_shots}"
            )


def write(result: workflow.Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("resiliency-figures"))
    parser.add_argument("--results", type=Path, default=Path("resiliency-results.json"))
    args = parser.parse_args()
    result = workflow.optimize()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
