"""Run and plot the original Gaussian bins and discrete price model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result, run


def print_report(result: Result) -> None:
    print(
        "Gaussian grid sizes:",
        len(result.gaussian_four.points),
        len(result.gaussian_one.points),
    )
    print("Binary Gaussian probabilities:", result.gaussian_one.probabilities)
    print("Sine and recurrence input rows:", len(result.sine), len(result.recurrence))
    print(
        "Retained words and named outputs:",
        result.nonzero_words,
        len(result.distribution),
    )
    print("Full-state and work errors:", result.state_error, result.workspace)
    print(
        "Price resources:",
        result.price_resources.n_qubits,
        result.price_resources.n_operations,
        result.price_resources.depth,
    )
    print("Probability mass:", sum(result.distribution.values()))


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    axes[0].bar(
        result.gaussian_four.points, result.gaussian_four.probabilities, width=0.20
    )
    axes[0].set(
        xlabel="Left endpoint", ylabel="Probability mass", title="Sixteen Gaussian bins"
    )
    for time, color in ((0, "#226987"), (1, "#b26d40")):
        pairs = sorted(
            (float(price), value)
            for (t, price), value in result.distribution.items()
            if t == time
        )
        axes[1].plot(
            [value for value, _ in pairs],
            [probability for _, probability in pairs],
            "o",
            color=color,
            label=f"Time {time}",
        )
    axes[1].set(
        xlabel="Fixed-point price",
        ylabel="Joint probability",
        title="Native time and price marginal",
    )
    axes[1].legend()
    figure.savefig(directory / "gaussian-and-price.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("brownian-chebyshev-figures")
    )
    args = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
