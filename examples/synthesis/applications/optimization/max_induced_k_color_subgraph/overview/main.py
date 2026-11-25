"""Run the complete native induced two-color subgraph workflow."""

import argparse

from .workflow import EDGES, model_row, run_workflow


def main() -> None:
    """Report actual observed feasibility separately from the exact reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=50)
    args = parser.parse_args()
    workflow = run_workflow(maxiter=args.maxiter)
    print("source_edges", EDGES)
    print("initial_CVaR", workflow.trace[0].cost)
    print("final_CVaR", workflow.optimizer.value)
    print("iterations", workflow.optimizer.iterations)
    print("actual_evaluations", workflow.optimizer.evaluations)
    print("converged", workflow.optimizer.converged)
    print("best_observed_feasible", workflow.best_feasible)
    for name, samples in (
        ("optimized", workflow.samples),
        ("uniform", workflow.uniform),
    ):
        print(f"{name}_observations", sum(row.count for row in samples))
        print(
            f"{name}_feasible_observations",
            sum(row.count for row in samples if row.feasible),
        )
    print("exact_bounded_reference", model_row(workflow.exact_reference))
    print("resources", workflow.program.report["resources"])
    print("temporary_workspace", workflow.program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
