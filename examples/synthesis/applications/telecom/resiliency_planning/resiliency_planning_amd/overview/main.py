"""Run the original CPU-bound 24-bit resiliency optimizer and final sampling."""

import argparse
from pathlib import Path

from examples.synthesis.applications.telecom.resiliency_planning.overview import (
    main as shared_main,
)
from examples.synthesis.applications.telecom.resiliency_planning.overview import (
    plots,
    workflow,
)

QUERIES, OBJECTIVE_SHOTS, FINAL_SHOTS = 1, 1000, 40_000


def run() -> workflow.Result:
    return workflow.optimize(
        queries=QUERIES, objective_shots=OBJECTIVE_SHOTS, final_shots=FINAL_SHOTS
    )


def report(result: workflow.Result) -> None:
    shared_main.report(result)
    valid_count = sum(count for _, count in result.feasible)
    print(f"Source displayed denominator=2000, actual final draws={FINAL_SHOTS}")
    print(
        f"Source denominator-scaled valid percentage={100 * valid_count / 2000}; actual valid percentage={100 * valid_count / FINAL_SHOTS}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("resiliency-amd-figures"))
    parser.add_argument(
        "--results", type=Path, default=Path("resiliency-amd-results.json")
    )
    args = parser.parse_args()
    result = run()
    report(result)
    plots.save(result, args.figures)
    shared_main.write(result, args.results)


if __name__ == "__main__":
    main()
