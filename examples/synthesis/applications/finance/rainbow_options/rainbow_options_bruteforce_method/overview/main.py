"""Run the original two-asset profile and display actual sampled evidence."""

from pathlib import Path

from .model import SOURCE_PRICE
from .workflow import Pricing, run_pricing

OUTPUT = Path("artifacts/development/rainbow-bruteforce")


def report(result: Pricing) -> None:
    print("Source affine offset:", result.model.offset)
    print("Price estimate:", result.price)
    print("Price interval:", result.interval)
    print("Independent discrete price:", result.model.exact_price)
    print(
        "Historical reference and assertion tolerance:",
        SOURCE_PRICE,
        result.source_tolerance,
    )
    print(
        "Actual indicator draws:",
        tuple((row.k, row.shots, row.successes) for row in result.estimate.iterations),
    )
    print("Full-state errors:", result.state_errors)


def save_figure(result: Pricing, output: Path = OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    axes[0].bar(result.model.grid, result.model.probabilities, width=0.7)
    axes[0].set(
        title="Original Gaussian bins", xlabel="Bin left edge", ylabel="Probability"
    )
    axes[1].plot(range(8), result.model.payoff_grid, marker="o")
    axes[1].set(
        title="Unscaled payoff on the selector",
        xlabel="Three-bit selector word",
        ylabel="Payoff before strike subtraction",
    )
    figure.tight_layout()
    path = output / "model.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    result = run_pricing()
    report(result)
    print(save_figure(result))


if __name__ == "__main__":
    main()
