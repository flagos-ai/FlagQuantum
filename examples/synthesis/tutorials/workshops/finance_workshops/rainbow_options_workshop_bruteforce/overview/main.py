"""Display the actual workshop estimate and its finite confidence interval."""

from pathlib import Path

from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.model import (
    SOURCE_PRICE,
)

from .workflow import SHOTS, Pricing, run_pricing

OUTPUT = Path("artifacts/development/rainbow-workshop")


def report(result: Pricing) -> None:
    print("Actual shots and successes:", SHOTS, result.successes)
    print("Observed indicator probability:", result.probability)
    print("Amplitude interval:", result.amplitude_interval)
    print("Price estimate:", result.price)
    print("Price interval:", result.price_interval)
    print("Independent discrete price:", result.model.exact_price)
    print(
        "Historical reference and assertion tolerance:",
        SOURCE_PRICE,
        result.source_tolerance,
    )
    print(
        "Historical check passed:",
        abs(result.price - SOURCE_PRICE) <= result.source_tolerance,
    )
    print("Full-state error:", result.state_error)


def save_figure(result: Pricing) -> Path:
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 2.4))
    low, high = result.price_interval
    axis.errorbar(
        result.price,
        0,
        xerr=[[result.price - low], [high - result.price]],
        fmt="o",
        capsize=6,
        label="20,000-shot estimate",
    )
    axis.axvline(
        result.model.exact_price,
        linestyle="--",
        color="0.4",
        label="Exact discrete reference",
    )
    axis.set(
        xlabel="Scaled option estimate",
        yticks=(),
        title="Rainbow workshop confidence interval",
    )
    axis.legend(loc="upper center")
    path = OUTPUT / "estimate.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    result = run_pricing()
    report(result)
    print(save_figure(result))


if __name__ == "__main__":
    main()
