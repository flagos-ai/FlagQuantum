"""Run both native autoencoder training routes, RESET decoding and detection."""

from .workflow import SHOTS, run_workflow


def main() -> None:
    """Print actual training, modal decoding and all anomaly scores."""
    report = run_workflow()
    for training in (report.swap, report.trash):
        print(
            training.cost,
            "minibatch L1 losses:",
            [step.loss for step in training.steps],
        )
    for decoded in report.decoded:
        print("Input/decoded:", decoded.data, decoded.modal_bits)
    for anomaly in report.anomalies:
        print("Input/cost/anomaly:", anomaly.data, anomaly.cost, anomaly.detected)
    print("Quantum evaluations/shots:", report.quantum_evaluations, SHOTS)


if __name__ == "__main__":
    main()
