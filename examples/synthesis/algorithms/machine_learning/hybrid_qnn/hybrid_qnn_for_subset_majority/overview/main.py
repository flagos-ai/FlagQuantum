"""Run the complete native subset-majority hybrid QNN example."""

from .workflow import SHOTS, run_workflow


def main() -> None:
    """Report actual sampled training, held-out accuracy and circuit resources."""
    workflow = run_workflow()
    for index, step in enumerate(workflow.steps, start=1):
        print("Minibatch/size/MSE:", index, step.size, step.loss)
    print("Test accuracy:", workflow.accuracy)
    print("Quantum evaluations/shots:", workflow.quantum_evaluations, SHOTS)
    print("Resources per bound circuit:", workflow.program.report["resources"])


if __name__ == "__main__":
    main()
