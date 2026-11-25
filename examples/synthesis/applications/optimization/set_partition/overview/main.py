"""Run the complete local set-partition optimization workflow."""

from .workflow import partitions, run_workflow


def main() -> None:
    """Print actual search, sampled partitions, reference and resource measurements."""
    workflow = run_workflow()
    print("source_weights", workflow.weights)
    print("initial_CVaR", workflow.trace[0].cost)
    print("final_CVaR", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("actual_evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed", workflow.best_observed)
    selected, other = partitions(workflow.best_observed.word, workflow.weights)
    print("observed_groups", selected, other)
    print("observed_sums", sum(selected), sum(other))
    print("optimized_shots", sum(row.count for row in workflow.samples))
    print("uniform_shots", sum(row.count for row in workflow.uniform))
    print(
        "exact_bounded_reference",
        workflow.exact_reference_word,
        partitions(workflow.exact_reference_word, workflow.weights),
    )
    print("resources", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
