"""Run the original seven-bit VaR search with fresh actual quantum samples."""

from pathlib import Path

from .model import discrete_quantile
from .plots import plot_distribution
from .workflow import run_risk


def run_example(seed: int = 6200) -> None:
    model, classical, measured, sampler = run_risk(seed)
    print("Independent discrete quantile:", discrete_quantile(model))
    print("Original exclusive-CDF classical search:", classical)
    print("Sample-only quantum search:", measured)
    print("Original function return:", measured.original_return)
    for threshold, result in sampler.estimates:
        print("Threshold:", threshold, "actual sampled estimate:", result)
    print("Independent full-state checks:", tuple(sampler.checks.values()))
    print("Actual structural programs:", len(sampler.programs))
    print(
        "The0.007search tolerance is tighter than the0.05estimation precision; convergence is not promised."
    )
    print(plot_distribution(model, Path("artifacts/development/value_at_risk")))


if __name__ == "__main__":
    run_example()
