"""Execute the credit-card fraud classification workflow with the source CSV."""

import argparse
from pathlib import Path

from . import model, plots, workflow


def report(result: workflow.Run) -> None:
    data = result.data
    score = result.score
    print("Source rows and class counts:", (data.rows, data.by_class))
    print(
        "Original split sizes:",
        tuple(
            len(group.labels)
            for group in (data.raw.train, data.raw.test, data.raw.predict)
        ),
    )
    print(
        "Input and PCA feature dimensions:",
        (data.raw.train.points.shape[1], data.features.train.points.shape[1]),
    )
    print("Complete overlap state errors:", result.verification)
    print(
        "Quantum test and prediction accuracy:",
        (score.test_accuracy, score.prediction_accuracy),
    )
    print(
        "Support vectors and sampled kernel entries:",
        (score.support_size, score.sampled_pairs),
    )
    print("Finite Bernoulli trial count:", workflow.trial_count(result))
    print("Classical test accuracies:", result.classical)
    print(
        "Test truth and predicted labels:",
        (tuple(map(int, data.raw.test.labels)), tuple(map(int, score.test_labels))),
    )
    print(
        "Prediction truth and predicted labels:",
        (tuple(map(int, data.raw.predict.labels)), tuple(map(int, score.predictions))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=model.DATA)
    parser.add_argument("--figures", type=Path, default=Path("credit-card-results"))
    args = parser.parse_args()
    result = workflow.run(args.data)
    report(result)
    for group, name in (
        (result.data.raw.train, "training"),
        (result.data.raw.test, "test"),
    ):
        plots.scatter(
            model.embed(group),
            group.labels,
            f"Credit-card {name} data",
            args.figures / f"{name}-tsne.png",
        )


if __name__ == "__main__":
    main()
