"""Plot every output word and source expected interval amplitudes."""

import math
from pathlib import Path

from matplotlib.figure import Figure


def plot_amplitudes(
    probabilities: tuple[float, ...],
    output: Path,
    name: str,
    *,
    interval: tuple[int, int] | None = None,
    rate: float = 0.5,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    if interval is not None:
        lower, upper = interval
        norm = math.fsum(math.exp(rate * x) for x in range(lower, upper + 1))
        axis.scatter(
            range(lower, upper + 1),
            [math.sqrt(math.exp(rate * x) / norm) for x in range(lower, upper + 1)],
            marker="+",
            s=100,
            label="Source expected interval",
        )
    axis.scatter(
        range(32),
        [math.sqrt(p) for p in probabilities],
        label="Measured probability square root",
        s=25,
    )
    axis.set(
        xlabel="x",
        ylabel="Square root of probability",
        title="Original five-bit exponential preparation",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    target = output / (name + ".png")
    figure.savefig(target, dpi=140)
    return target
