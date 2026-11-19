"""Report the native Rainbow integration estimate and reference law."""

import math
from pathlib import Path

from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.overview.model import (
    SOURCE_PRICE,
)

from .workflow import Pricing, run_pricing

OUTPUT = Path("artifacts/development/rainbow-integration")


def report(result: Pricing) -> None:
    model = result.model
    print(
        "Source affine offset and threshold:",
        model.asset_model.offset,
        model.asset_model.threshold,
    )
    print("Price offset and scale:", model.price_offset, model.price_scale)
    print("Price estimate:", result.price)
    print("Price interval:", result.interval)
    print("Independent fixed-point price:", model.asset_model.exact_price)
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


def save_figure(result: Pricing) -> Path:
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    words = range(len(result.model.weights))
    cumulative = [math.fsum(result.model.weights[: word + 1]) for word in words]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(words, result.model.weights)
    axes[0].set(
        title="Exponential reference distribution",
        xlabel="Reference word",
        ylabel="Probability",
    )
    axes[1].step(words, cumulative, where="post")
    axes[1].set(
        title="Inclusive reference comparison",
        xlabel="Score word",
        ylabel="Probability of reference ≤ score",
    )
    figure.tight_layout()
    path = OUTPUT / "integrator.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    result = run_pricing()
    report(result)
    print(save_figure(result))


if __name__ == "__main__":
    main()
