"""Show sampled and exact probabilities for the three source preparations."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from .workflow import SHOTS, CompressionResult


def _pyplot() -> Any:
    return import_module("matplotlib.pyplot")


def save_distributions(results: Sequence[CompressionResult], path: Path) -> Path:
    """Compare each native readout with its actual 2,048-shot sample."""
    pyplot = _pyplot()
    fig, axes = pyplot.subplots(len(results), 1, figsize=(9, 7.5), sharey=True)
    for axis, result in zip(axes, results, strict=True):
        # Keep every observed or exact nonzero word, including out-of-subspace words.
        words = [
            i
            for i, (p, count) in enumerate(
                zip(result.probabilities, result.counts, strict=True)
            )
            if p > 1e-12 or count
        ]
        axis.bar(
            [i - 0.18 for i in words],
            [result.probabilities[i] for i in words],
            0.36,
            label="Exact",
        )
        axis.bar(
            [i + 0.18 for i in words],
            [result.counts[i] / SHOTS for i in words],
            0.36,
            label="Sampled",
        )
        axis.set(
            title=result.scenario.replace("_", " ").title(),
            ylabel="Probability",
            ylim=(0, 1.08),
            xlim=(-0.6, 7.6),
            xticks=words,
        )
        axis.legend()
    axes[-1].set_xlabel("Named output word")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    pyplot.close(fig)
    return path
