"""Execute all nine source CFL events and report sampled spatial marginals."""

from pathlib import Path

from .model import SHOTS, active_magnitudes, schedule
from .plots import save_marginal, save_reflection, save_schedule
from .workflow import run_workflow

FOLDER = Path("artifacts/development/qlbm")


def main() -> None:
    workflow = run_workflow()
    print("CFL events and final time:", len(schedule()), str(schedule()[-1][0]))
    print("Active magnitudes:", active_magnitudes())
    print("Actual shots:", sum(workflow.counts.values()))
    assert sum(workflow.counts.values()) == SHOTS
    print("Position marginal:", workflow.position.tolist())
    print("Magnitude marginal:", workflow.magnitude.tolist())
    print("Native resources:", workflow.program.report["resources"])
    print(save_schedule(FOLDER / "schedule.png"))
    print(save_marginal(workflow.position, "position", FOLDER / "position.png"))
    print(save_marginal(workflow.magnitude, "magnitude", FOLDER / "magnitude.png"))
    print(save_reflection(FOLDER / "reflection.png"))


if __name__ == "__main__":
    main()
