"""Run the complete original-sized kidney matching optimization."""

from pathlib import Path

from .plots import plot_results
from .workflow import run_workflow


def run_example() -> None:
    print("Initial classical optimization: not solved; source solver is commented out.")
    workflow = run_workflow()
    print("Actual native objective evaluations:", len(workflow.trace))
    print("Top ten observations:", workflow.samples[:10])
    print("Best observed assignment:", workflow.best_observed)
    print("Best observed feasible assignment:", workflow.best_observed_feasible)
    print("Separate exhaustive matching reference:", workflow.reference)
    best = workflow.best_observed.assignment
    print("Source printed score (-penalized cost):", -best.penalized_cost)
    print("Actual compatibility score:", best.score)
    print("Conflict penalty:", 2 * best.conflict_pairs)
    print("Donor/patient table:")
    for donor, row in enumerate(best.rows, start=1):
        print(f"donor{donor}:", row)
    print(plot_results(workflow, Path("artifacts/development/kidney-exchange")))


if __name__ == "__main__":
    run_example()
