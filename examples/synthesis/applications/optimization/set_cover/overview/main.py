"""Run the full original Set Cover comparison from the command line."""

from pathlib import Path

from .model import Model
from .plots import draw_histogram, draw_trace
from .workflow import run_workflow


def selected_sets(model: Model, word: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset for index, subset in enumerate(model.sets) if word & (1 << index)
    )


def run_example() -> None:
    workflow = run_workflow()
    best = workflow.optimized[0]
    print("Original subsets:", workflow.model.sets)
    print("Selected resources:", workflow.program.report["resources"])
    print("Actual iterations:", workflow.optimizer.iterations)
    print("Actual objective evaluations:", len(workflow.trace))
    print(
        "Initial and selected CVaR:",
        workflow.trace[0].cost,
        min(row.cost for row in workflow.trace),
    )
    print("Converged:", workflow.optimizer.converged)
    print(
        "Optimized and uniform draws:",
        *(
            sum(row.count for row in values)
            for values in (workflow.optimized, workflow.uniform)
        )
    )
    print("Five lowest observed penalized costs:", workflow.optimized[:5])
    print("Best actually observed:", best)
    print("Observed sets:", selected_sets(workflow.model, best.decisions))
    print("Exact minimum words:", workflow.classical)
    print("One exact solution:", selected_sets(workflow.model, workflow.classical[0]))
    path = Path("artifacts/development/set-cover")
    print(draw_trace(workflow, path / "cost-trace.png"))
    print(draw_histogram(workflow, path / "cost-histogram.png"))


if __name__ == "__main__":
    run_example()
