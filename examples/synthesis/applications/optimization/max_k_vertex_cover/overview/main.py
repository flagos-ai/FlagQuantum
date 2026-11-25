"""Run the complete native maximum-k-vertex-cover Notebook workflow."""

from examples.synthesis.graph_optimization import selected_nodes

from .workflow import model_table, run_workflow, source_edges


def main() -> None:
    """Report actual optimization, feasible observations and a separate reference."""
    workflow = run_workflow()
    print("source_graph_edges", source_edges())
    print("K", 5)
    print("initial_CVaR", workflow.trace[0].cost)
    print("final_CVaR", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("actual_evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed_feasible", workflow.best_feasible)
    if workflow.best_feasible is not None:
        best = workflow.best_feasible
        print("observed_nodes", selected_nodes(best.word, 10))
        print("covered_edges", len(source_edges()) - best.uncovered)
    print("optimized_shots", sum(row.count for row in workflow.samples))
    print("uniform_shots", sum(row.count for row in workflow.uniform))
    print(
        "feasible_optimized_observations",
        sum(row.count for row in workflow.samples if row.feasible),
    )
    print(
        "feasible_uniform_observations",
        sum(row.count for row in workflow.uniform if row.feasible),
    )
    print(
        "exact_bounded_reference_nodes",
        selected_nodes(workflow.exact_reference_word, 10),
    )
    print(
        "exact_bounded_reference_uncovered",
        model_table()[workflow.exact_reference_word][0],
    )
    print("resources", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
