"""Run original-sized fresh option preparation and count-only adaptive pricing."""

from pathlib import Path

from .plots import plot_distribution
from .workflow import run_pricing


def run_example(
    *, max_width: int = 20, seed: int = 6100, source_tolerance_factor: int = 5
) -> None:
    result, sampler = run_pricing(
        max_width=max_width, seed=seed, source_tolerance_factor=source_tolerance_factor
    )
    print("Fresh grid, probabilities and payoff angles:", result.model)
    print("Measured payoff:", result.price)
    print("Price confidence interval:", result.interval)
    print("Independent expected payoff:", result.model.expected_payoff)
    print(
        "Local count-only estimator: epsilon0.05,alpha0.01,maximum64steps,k<=16,batch256"
    )
    for step in result.estimate.iterations:
        print(step)
    print("Full-state independent checks:", result.checks)
    print("Saved programs:", len(sampler.programs))
    print(
        "Actual resources:",
        tuple(program.report["resources"] for program in result.programs),
    )
    print(plot_distribution(result.model, Path("artifacts/development/option_pricing")))


if __name__ == "__main__":
    run_example()
