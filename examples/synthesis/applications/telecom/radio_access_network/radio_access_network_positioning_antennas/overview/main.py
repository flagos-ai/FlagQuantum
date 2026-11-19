"""Run the complete fresh-data native radio-access-network source workflow."""

from pathlib import Path

from .plots import plot_results
from .workflow import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    print("Fresh source seed0 coefficients:", workflow.coefficients)
    print("Source maximizes consumption score; the objective sign is retained.")
    print("Actual native objective evaluations:", len(workflow.trace))
    print("Top five actual observations:", workflow.samples[:5])
    best = workflow.samples[0]
    print("Selected locations:", best.assignment.selected)
    print("Consumption score:", best.assignment.score)
    print("Source penalized cost:", best.cost)
    print(
        "Decision feasibility:",
        best.assignment.feasible,
        "; residual penalty:",
        best.penalty,
    )
    print(plot_results(workflow, Path("artifacts/development/radio-access-network")))


if __name__ == "__main__":
    run_example()
