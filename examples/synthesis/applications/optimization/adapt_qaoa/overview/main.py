"""Run the source ordinary and adaptive quantum algorithms locally."""

from pathlib import Path

from .model import costs
from .plots import plot_cost
from .resources import source_basis_metrics
from .workflow import run_workflow


def run_example() -> None:
    output = Path("artifacts/development/adapt-qaoa")
    result = run_workflow()
    print("Independent exact minimum energy:", float(costs().min()))
    print(
        "Ordinary source basis metrics:", source_basis_metrics(result.ordinary.program)
    )
    for row in result.layers:
        print(
            "Adaptive layer mixers:",
            row.mixers,
            "scaled gradient norm:",
            row.gradient_norm,
        )
    for label, optimization, samples in (
        ("Ordinary", result.ordinary, result.ordinary_samples),
        ("Adaptive", result.layers[-1].optimization, result.adaptive_samples),
    ):
        print(
            label,
            "queries:",
            len(optimization.trace),
            "budget stop:",
            optimization.budget_stop,
        )
        print(label, "angles:", optimization.parameters)
        print(label, "draws:", sum(row.count for row in samples))
        print(label, "ten lowest observed energies:", samples[:10])
        print(
            plot_cost(optimization, label + " QAOA", output / (label.lower() + ".png"))
        )


if __name__ == "__main__":
    run_example()
