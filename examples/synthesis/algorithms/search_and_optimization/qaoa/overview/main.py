"""Run both native QAOA tutorial sections with actual optimization and readout."""

from .workflow import Case, run_workflow


def main() -> None:
    """Report actual bounded search and observed feasible candidates."""
    cases: tuple[Case, ...] = ("maxcut", "knapsack")
    for case in cases:
        workflow = run_workflow(case)
        print("section", case)
        print("initial_cost", workflow.trace[0].cost)
        print("final_cost", workflow.optimizer.value)
        print("iterations", workflow.optimizer.iterations)
        print("actual_evaluations", workflow.optimizer.evaluations)
        print("converged", workflow.optimizer.converged)
        print("best_observed_feasible", workflow.best_feasible)
        print("resources", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
