"""Run all source QCE part II models, including the two solution replays."""

from .model import Model, feasible_value
from .presentation import sample_table
from .verification import verify_all
from .workflow import optimize, statistics


def run_example() -> None:
    print("Exercise 1 feasible values:", feasible_value(3, 2), feasible_value(3, 3))
    for check in verify_all():
        print(check)
    stages: tuple[Model, ...] = ("maxcut", "penalty", "feasible", "penalty", "feasible")
    for index, model in enumerate(stages):
        run = optimize(model, seed=1970 + 2 * index)
        print(model, "Actual objective evaluations:", len(run.evaluations))
        print("Selected last objective:", run.selected.cost, run.termination)
        if model != "maxcut":
            print(statistics(run.statistics_samples))
        print(sample_table(run))


if __name__ == "__main__":
    run_example()
