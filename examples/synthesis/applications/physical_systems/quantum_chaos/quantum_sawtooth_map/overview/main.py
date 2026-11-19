"""Run all four original kicked-map times with actual local samples."""

from .workflow import historical_observations, run_workflow, verify


def run_example() -> None:
    one_step = verify(1)
    print("Actual one-step CX/U3 resources:", one_step.resources)
    for row in run_workflow():
        print("T=", row.evolution.kicks, row.evolution.resources)
        print("Fresh local samples:", row.observations)
        print("Historical upstream IonQ:", historical_observations(row.evolution.kicks))


if __name__ == "__main__":
    run_example()
