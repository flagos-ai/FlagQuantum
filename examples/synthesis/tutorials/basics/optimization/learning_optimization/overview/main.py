"""Run the bounded integer optimization tutorial locally."""

from .workflow import run_workflow


def main() -> None:
    """Report the actual search, convergence flag, samples, and resources."""
    workflow = run_workflow()
    print("minimize 3*x + 2*y; 3*x + y >= 2; x,y in {0,1,2,3}")
    print("initial_expectation", workflow.trace[0].expectation)
    print("final_expectation", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed_feasible", workflow.best_feasible)
    print(
        "feasible_samples", sum(row.count for row in workflow.samples if row.feasible)
    )
    print("resources", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
