"""Run all original portfolio workshop scenarios with native synthesis."""

from ...portfolio_workflow import (
    EQUALITY_SCENARIOS,
    bindings,
    classical_optima,
    run_workflow,
    verify,
)


def run_example() -> None:
    for index, scenario in enumerate(EQUALITY_SCENARIOS):
        for parameters in bindings(scenario):
            print(verify(scenario, parameters))
        workflow = run_workflow(scenario, seed=1940 + index)
        print(scenario)
        print(workflow.termination, workflow.converged, len(workflow.trace))
        print(workflow.trace[0].cost, workflow.selected.cost)
        print(workflow.best_observed)
        print("Classical minima:", classical_optima(scenario))
        print("Actual shots:", sum(row.count for row in workflow.samples))
        print(verify(scenario, workflow.selected.parameters))


if __name__ == "__main__":
    run_example()
