"""Run both original four-qubit circuits and compare actual samples."""

from pathlib import Path

from .plots import distributions, objective_trace
from .workflow import SHOTS, Run, optimize


def print_result(label: str, result: Run) -> None:
    counts = sum(row.count for row in result.observations)
    print(label, "objective queries:", len(result.trace))
    print(label, "actual objective draws:", SHOTS * len(result.trace))
    print(label, "best sampled objective:", min(row.cost for row in result.trace))
    print(label, "budget stop:", result.budget_reached, "converged:", result.converged)
    print(label, "actual final draws:", counts)
    print(
        label,
        "one-hot final draws:",
        sum(row.count for row in result.observations if row.one_hot),
    )
    print(label, "resources:", result.program.report["resources"])
    print(label, "observations:", result.observations)


def run_example() -> None:
    ordinary, grover = optimize(grover=False), optimize(grover=True)
    print_result("Ordinary", ordinary)
    print_result("Grover", grover)
    folder = Path("artifacts/development/grover-mixer-qaoa")
    print(
        objective_trace(
            ordinary, "Ordinary QAOA objective", folder / "ordinary-trace.png"
        )
    )
    print(
        objective_trace(grover, "Grover-mixer objective", folder / "grover-trace.png")
    )
    print(distributions(ordinary, grover, folder / "sample-comparison.png"))


if __name__ == "__main__":
    run_example()
