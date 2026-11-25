"""Original log-normal grid and option payoff using freshly computed model data."""

from pathlib import Path

import matplotlib.pyplot as plt

from .model import OptionModel


def plot_distribution(model: OptionModel, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    figure, probability = plt.subplots(figsize=(8, 4.5))
    payoff = probability.twinx()
    probability.plot(
        model.grid, model.probabilities, "go-", label="Discrete probability"
    )
    probability.set(
        xlabel="Asset value at maturity",
        ylabel="Probability",
        title="Fresh log-normal distribution and call payoff",
    )
    payoff.plot(
        model.grid,
        [max(price - model.strike, 0) for price in model.grid],
        "r-",
        label="Call payoff",
    )
    payoff.set_ylabel("Payoff")
    probability.grid(alpha=0.3)
    figure.tight_layout()
    path = directory / "distribution_payoff.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
