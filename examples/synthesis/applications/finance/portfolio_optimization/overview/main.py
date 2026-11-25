"""Run the complete original-size portfolio workflow locally."""

from ..main import DEFAULT_PARAMS
from .operations import verify
from .workflow import classical_optima, run_workflow


def run_example() -> None:
    for parameters in (DEFAULT_PARAMS, (-0.023, 0.041, -0.017, 0.052, -0.033, 0.061)):
        print(verify(parameters))
    workflow = run_workflow()
    print("Iterations:", workflow.optimizer.iterations)
    print("Objective evaluations:", workflow.optimizer.evaluations)
    print("Converged:", workflow.optimizer.converged)
    print("Initial CVaR:", workflow.trace[0].cvar)
    print("Selected CVaR:", workflow.optimizer.value)
    print("Best feasible observed holding:", workflow.best_observed)
    print("Independent classical optima:", classical_optima())
    print(
        "Optimized/uniform shots:",
        sum(row.count for row in workflow.optimized),
        sum(row.count for row in workflow.uniform),
    )
    print(verify(tuple(float(v) for v in workflow.optimizer.parameters)))


if __name__ == "__main__":
    run_example()
