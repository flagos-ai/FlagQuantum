"""Run the complete original-sized native facility-location Notebook workflow."""

from pathlib import Path

from .model import FACILITIES
from .plots import plot_results
from .workflow import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    print(
        "Three facilities, four customers, exactly two used facilities; twelve decisions."
    )
    print("Actual objective evaluations:", len(workflow.trace))
    print("Top five actual observations:", workflow.samples[:5])
    best = workflow.best_observed.assignment
    print("Best observed raw cost:", best.raw_cost)
    print("Source penalized cost:", best.cost)
    print("Feasible:", best.feasible, "; penalty:", best.penalty)
    print("Used facilities:", tuple(FACILITIES[i] for i in best.opened))
    for name, row in zip(FACILITIES, best.rows, strict=True):
        print(name, row)
    print(
        "Separate exhaustive constrained solution, solved after sampling:",
        workflow.reference,
    )
    print(plot_results(workflow, Path("artifacts/development/facility-location")))


if __name__ == "__main__":
    run_example()
