"""Run the native graph optimization workflow and report actual results."""

from .workflow import run_workflow


def main() -> None:
    """Report bounded optimization, observed feasible answer, reference and costs."""
    workflow = run_workflow()
    print("initial_CVaR", workflow.trace[0].cost)
    print("final_CVaR", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("actual_evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed_feasible", workflow.best_feasible)
    print("exact_bounded_reference", workflow.exact_nodes)
    print("resources", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
