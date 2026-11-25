"""Run the complete source-sized native electric-grid optimization workflow."""

from pathlib import Path

from .plots import plot_results
from .workflow import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    print(
        "Original18bits:12decisions plus six bounded-slack bits; four layers,100iterations."
    )
    print("Actual native evaluations:", len(workflow.trace))
    print("Top five joint observations:", workflow.samples[:5])
    best = workflow.best_observed
    print("Best observed raw cost:", best.assignment.raw_cost)
    print("Source reported penalized cost:", best.penalized_cost)
    print(
        "Decision feasibility:",
        best.assignment.feasible,
        "; residual penalty:",
        best.penalty,
    )
    print("Source/consumer table:", best.assignment.rows)
    print(
        "Separate exact constrained solution, solved after sampling:",
        workflow.reference,
    )
    print(plot_results(workflow, Path("artifacts/development/electric-grid")))


if __name__ == "__main__":
    run_example()
