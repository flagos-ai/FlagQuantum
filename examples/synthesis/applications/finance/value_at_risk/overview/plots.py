"""Render the full128point distribution and original7percent quantile."""

from pathlib import Path

from matplotlib.figure import Figure

from .model import LossModel, discrete_quantile


def plot_distribution(model: LossModel, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    axis.plot(
        model.grid, model.probabilities, "go-", markersize=2.5, label="Probability"
    )
    _, quantile = discrete_quantile(model)
    axis.axvline(quantile, color="red", linestyle="--", label="VaR at7%")
    axis.set(
        xlabel="Asset value",
        ylabel="Probability",
        title="Fresh128point log-normal distribution",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    path = output / "value-at-risk.png"
    figure.savefig(path, dpi=140)
    return path
