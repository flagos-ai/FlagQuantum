"""Run the complete native integer linear programming Notebook workflow."""

from .workflow import SOURCE_A, SOURCE_B, SOURCE_C, model_row, run_workflow


def main() -> None:
    """Report original-model feasibility separately from the retained quantum cost."""
    workflow = run_workflow()
    print("original_A", SOURCE_A)
    print("original_b", SOURCE_B)
    print("original_c", SOURCE_C)
    print("integer_bounds", (0, 3))
    print("initial_CVaR", workflow.trace[0].cost)
    print("final_CVaR", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("actual_evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed_feasible", workflow.best_feasible)
    if workflow.best_feasible is not None:
        best = workflow.best_feasible
        print("original_assignment", best.assignment[:3])
        print("modeled_slack_outputs", best.assignment[3:])
        print("original_objective", best.objective)
        print("all_constraint_left_sides", best.lhs)
        print("retained_quantum_cost", best.retained_quantum_cost)
    for name, samples in (
        ("optimized", workflow.samples),
        ("uniform", workflow.uniform),
    ):
        print(f"{name}_observations", sum(row.count for row in samples))
        print(
            f"{name}_feasible_observations",
            sum(row.count for row in samples if row.feasible),
        )
    reference = model_row((*workflow.exact_reference, 0, 0))
    print("exact_bounded_original_assignment", workflow.exact_reference)
    print("exact_bounded_original_objective", reference.objective)
    print("resources", workflow.program.report["resources"])
    print("temporary_workspace", workflow.program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
