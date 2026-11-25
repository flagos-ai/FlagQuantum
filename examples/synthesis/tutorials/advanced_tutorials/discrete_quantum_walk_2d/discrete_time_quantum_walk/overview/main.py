"""Run the full-size ring and torus Notebook's three actual observation batches."""

from .workflow import run_workflow


def run_example() -> None:
    for evolution, observations in run_workflow():
        print("Dimension:", evolution.dimension, "Steps:", evolution.steps)
        print("Absolute full-state error:", evolution.maximum_error)
        print("Dirty compiler workspace:", evolution.dirty_workspace_probability)
        print(evolution.program.report["resources"])
        print("Actual draws:", sum(row.count for row in observations))
        print(observations)


if __name__ == "__main__":
    run_example()
