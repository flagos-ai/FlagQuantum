"""Run all original forty-node QAOA communities with fresh native programs."""

from pathlib import Path

from .plots import plot_solution
from .workflow import run_workflow


def run_example() -> None:
    workflow = run_workflow()
    print(
        "Original graph:",
        len(workflow.graph),
        "nodes and",
        len(workflow.graph.edges),
        "edges",
    )
    print("Initial communities:", workflow.initial_groups)
    print("Contraction sizes:", workflow.contraction_sizes)
    for index, result in enumerate(workflow.runs):
        print(
            "Subproblem",
            index,
            "nodes",
            result.nodes,
            "evaluations",
            len(result.trace),
            "actual samples",
            sum(row.count for row in result.observations),
            "budget reached",
            result.budget_reached,
            "resources",
            result.resources,
        )
    print("Observed full assignment:", dict(enumerate(workflow.assignment)))
    print("CUT OF QAOA SQUARE:", workflow.cut)
    print("Greedy one-exchange cut:", workflow.greedy_cut)
    print("Random partition cut:", workflow.random_cut)
    print(
        plot_solution(
            workflow, Path("artifacts/development/qaoa-in-qaoa/observed-cut.png")
        )
    )


if __name__ == "__main__":
    run_example()
