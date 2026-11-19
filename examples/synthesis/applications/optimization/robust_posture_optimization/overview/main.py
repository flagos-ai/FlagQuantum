"""Run native posture optimization and report independently resampled positions."""

from pathlib import Path

from .measurements import TARGET
from .plots import plot_trace
from .workflow import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    print("Actual objective evaluations:", len(workflow.trace))
    print("Selected measured cost:", workflow.selected.estimate.cost)
    print("Termination:", workflow.termination, "; converged:", workflow.converged)
    print("Final 1,000 source Y-basis counts:", workflow.final_y.counts)
    for index, value in enumerate(workflow.positions, 1):
        print("Independent position estimate", index, value.position)
    print("Target arm position:", TARGET)
    print(
        plot_trace(
            workflow, Path("artifacts/development/robust-posture/cost-convergence.png")
        )
    )


if __name__ == "__main__":
    run_example()
