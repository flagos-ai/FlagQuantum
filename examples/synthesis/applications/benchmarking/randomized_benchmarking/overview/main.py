"""Run every original-length local sequence/inverse experiment and tagged analysis."""

from pathlib import Path

from .plots import plot_experiment
from .workflow import run_experiment


def run_example() -> None:
    experiment = run_experiment()
    for trial in experiment.trials:
        print(trial)
    print("Execution batch sizes:", experiment.batch_sizes)
    print(
        "Source repeated-length label mismatches:", experiment.source_label_mismatches
    )
    print("The plot uses each executed job's own sequence length and simulator mode.")
    print(
        "These ideal local observations cannot identify noisy-device RB decay or gate error."
    )
    print(
        plot_experiment(
            experiment, Path("artifacts/development/randomized_benchmarking")
        )
    )


if __name__ == "__main__":
    run_example()
