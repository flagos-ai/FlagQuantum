"""Run the original four-layer native MaxCut Workshop locally."""

from ..comparisons.circuit import BINDINGS
from .workflow import (
    classical_optima,
    cut_cost,
    initial_parameters,
    run_workflow,
    verify,
)


def run_example() -> None:
    for parameters in (initial_parameters(), *BINDINGS):
        print(verify(parameters))
    workflow = run_workflow()
    print("Termination:", workflow.termination, "Converged:", workflow.converged)
    print("Actual objective evaluations:", len(workflow.trace))
    print("Initial and selected means:", workflow.trace[0].cost, workflow.selected.cost)
    print("Actual shots:", sum(row.count for row in workflow.samples))
    print("Best observed cut:", -workflow.best_observed.cost, workflow.best_observed)
    print("Classical maximum cut:", -cut_cost(classical_optima()[0]))
    print(verify(workflow.selected.parameters))


if __name__ == "__main__":
    run_example()
