"""Run both native QML guide workflows on the local CPU."""

from .workflow import run_workflow


def main() -> None:
    """Report actual optimizer, training and synthesis results."""
    workflow = run_workflow()
    result = workflow.variational.optimizer
    print("VQE energy:", result.value)
    print("VQE angles:", result.parameters)
    print(
        "VQE iterations/evaluations/converged:",
        result.iterations,
        result.evaluations,
        result.converged,
    )
    print("VQE resources:", workflow.variational.program.report["resources"])
    for epoch, step in enumerate(workflow.training.steps):
        print("NOT epoch:", epoch, step)
    print(
        "NOT weight/predictions/accuracy:",
        workflow.training.weight,
        workflow.training.predictions,
        workflow.training.accuracy,
    )
    print("NOT resources:", workflow.training.program.report["resources"])


if __name__ == "__main__":
    main()
