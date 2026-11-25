"""Run every original-sized random-circuit trial with fresh matrices and readout."""

from pathlib import Path

from .plots import plot_experiment
from .workflow import run_experiment, source_round_significant


def run_example() -> None:
    experiment = run_experiment()
    print("Saved native quantum programs:", len(experiment.programs))
    for row in experiment.sizes:
        print(
            row.size,
            row.observed_heavy_probability,
            source_round_significant(row.source_standard_deviation),
            row.lower_bound,
            row.passes_source_threshold,
        )
    print("Source observed-count volume diagnostic:", experiment.source_volume)
    print(
        "Maximum absolute state error:", max(t.state_error for t in experiment.trials)
    )
    print(
        "Maximum named probability error:",
        max(t.named_probability_error for t in experiment.trials),
    )
    print(
        "Actual CX per circuit:",
        tuple(
            (t.size, t.index, t.resources.operation_counts.get("cx", 0))
            for t in experiment.trials
        ),
    )
    plot_experiment(experiment, Path("artifacts/development/quantum_volume"))


if __name__ == "__main__":
    run_example()
