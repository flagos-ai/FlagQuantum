"""Execute both original likelihood policies and report actual observations."""

from pathlib import Path

from .plots import save_posterior
from .workflow import ANNEAL_STEPS, NEIGHBORS, run_workflow

FOLDER = Path("artifacts/development/quantum-likelihood")


def main() -> None:
    workflow = run_workflow()
    print("Annealing steps and neighbors:", ANNEAL_STEPS, NEIGHBORS)
    for mode, history, heading in (
        ("standard", workflow.standard, "Standard particle selection"),
        ("optimized", workflow.optimized, "Conditional-entropy selection"),
    ):
        print(f"{mode} actual measurements:", len(history))
        print(f"{mode} final posterior:", history[-1].posterior)
        print(f"{mode} observed outcomes:", tuple(row.outcome for row in history))
        print(save_posterior(history, heading, FOLDER / f"{mode}.png"))


if __name__ == "__main__":
    main()
