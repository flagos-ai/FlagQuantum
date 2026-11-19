"""Train and compare four support vector classifiers at the original sizes."""

import argparse
from pathlib import Path

from . import plots, workflow


def report_score(label: str, score: workflow.Score) -> None:
    print(
        f"{label}:",
        score.test_accuracy,
        score.prediction_accuracy,
        score.support_size,
        score.sampled_pairs,
    )
    print(f"{label} predictions:", tuple(map(int, score.predictions)))


def print_report(data: workflow.Result) -> None:
    for label, split in (("Synthetic", data.synthetic), ("Iris", data.iris)):
        print(
            f"{label} split sizes:",
            tuple(
                len(group.labels) for group in (split.train, split.test, split.predict)
            ),
        )
        print(f"{label} ground truth:", tuple(map(int, split.predict.labels)))
    report_score("Synthetic Bloch", data.synthetic_bloch)
    report_score("Iris Pauli", data.iris_pauli)
    report_score("Iris Bloch", data.iris_bloch)
    report_score("Iris RBF", data.iris_rbf)
    pairs = sum(
        score.sampled_pairs
        for score in (data.synthetic_bloch, data.iris_pauli, data.iris_bloch)
    )
    print("Finite Bernoulli trial count:", pairs * workflow.SHOTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("qsvm-results"))
    args = parser.parse_args()
    data = workflow.run()
    print_report(data)
    plots.synthetic(data.synthetic, args.figures)
    plots.prediction(data.synthetic.predict, data.synthetic_bloch, args.figures)
    plots.iris(data.iris, args.figures)


if __name__ == "__main__":
    main()
