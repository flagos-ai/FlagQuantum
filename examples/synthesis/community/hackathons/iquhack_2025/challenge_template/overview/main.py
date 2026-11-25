"""Run both original iQuHack starter stages and compare a local reference."""

import argparse
from pathlib import Path

from .workflow import Comparison, Stage, build, compare, resources


def report_costs(stage: Stage, number: int) -> None:
    print(
        f"Stage {number} starter width, gates, depth and CX:", resources(stage.starter)
    )
    print(
        f"Stage {number} reference width, gates, depth and CX:",
        resources(stage.reference),
    )


def report_result(result: Comparison) -> None:
    print("Observed starter signed sample:", result.starter_sample)
    print("Starter signed-grid MSE:", result.starter_mse)
    print("Reference signed-grid MSE:", result.reference_mse)
    print(
        "Starter and reference absolute state errors:",
        (result.absolute_starter_error, result.absolute_reference_error),
    )


def plot(result: Comparison, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
    axis.plot(
        result.grid,
        result.gaussian,
        color="#bf562d",
        linewidth=2,
        label="Gaussian target",
    )
    axis.scatter(
        result.grid,
        result.reference,
        color="#225980",
        alpha=0.65,
        s=8,
        label="Native reference state",
    )
    axis.plot(
        result.grid,
        result.starter,
        color="#637283",
        linestyle="--",
        label="Original uniform starter",
    )
    axis.set(
        xlabel="Signed fixed-point coordinate",
        ylabel="Probability",
        title="iQuHack Gaussian starter and reference",
    )
    axis.legend(frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "starter-and-reference.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("gaussian-challenge-figures")
    )
    options = parser.parse_args()
    stage = build(1)
    result = compare(stage)
    report_result(result)
    report_costs(stage, 1)
    plot(result, options.figures)
    report_costs(build(2), 2)
    print("Team export:", "disabled")


if __name__ == "__main__":
    main()
