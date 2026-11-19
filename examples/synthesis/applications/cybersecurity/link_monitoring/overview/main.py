"""Run original-size link monitoring with the source's fixed resource node."""

from .reference import verify
from .workflow import classical_optima, run_workflow


def run_example() -> None:
    for parameters in ((0.013, 0.026), (-0.019, 0.046)):
        print(verify(parameters))
    workflow = run_workflow()
    print("Native iterations:", workflow.optimizer.iterations)
    print("Actual objective executions:", workflow.optimizer.evaluations)
    print("Converged:", workflow.optimizer.converged)
    print(
        "Initial and selected CVaR:", workflow.trace[0].cost, workflow.optimizer.value
    )
    print("Actual shots:", sum(row.count for row in workflow.samples))
    print("Observed minimum feasible cover:", workflow.best_feasible)
    print("Independent classical minima:", classical_optima())
    print(verify(tuple(float(v) for v in workflow.optimizer.parameters)))


if __name__ == "__main__":
    run_example()
