"""Run the complete original-size patch selection workflow locally."""

from ..comparisons.circuit import BINDINGS
from .operations import verify
from .workflow import (
    dual_graph,
    model_tables,
    remaining_graph,
    run_workflow,
    source_graph,
)


def run_example() -> None:
    graph, dual = source_graph(), dual_graph()
    print(
        "Original graph:",
        graph.number_of_nodes(),
        "nodes,",
        graph.number_of_edges(),
        "edges",
    )
    print(
        "Resource conflict graph:",
        dual.number_of_nodes(),
        "nodes,",
        dual.number_of_edges(),
        "edges",
    )
    costs, feasible = model_tables()
    print("Complete binary assignments:", len(costs), "Feasible:", len(feasible))
    for parameters in BINDINGS:
        print(verify(parameters))
    workflow = run_workflow()
    print("Iterations:", workflow.optimizer.iterations)
    print("Objective evaluations:", workflow.optimizer.evaluations)
    print("Converged:", workflow.optimizer.converged)
    print("Initial CVaR:", workflow.trace[0].cost)
    print("Selected CVaR:", workflow.optimizer.value)
    print(
        "Actual optimized/uniform shots:",
        sum(row.count for row in workflow.optimized),
        sum(row.count for row in workflow.uniform),
    )
    best = workflow.best_feasible
    print("Best observed feasible patch set:", best)
    if best is not None:
        remaining = remaining_graph(best.nodes)
        print(
            "All shared-vulnerability resource pairs removed:",
            all(remaining.degree(v) <= 1 for v in "ABCDEFG"),
        )
    print("Independent classical minimum:", min(costs[word] for word in feasible))
    print(verify(tuple(float(v) for v in workflow.optimizer.parameters)))


if __name__ == "__main__":
    run_example()
