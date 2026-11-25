"""Run the native source-size rectangle packing workflow."""

from pathlib import Path

from .plots import save_costs, save_packing, save_trace
from .workflow import SHOTS, run_workflow

FOLDER = Path("artifacts/development/rectangle-packing")


def main() -> None:
    result = run_workflow()
    print(
        "Placement variables and Ising terms:",
        len(result.model.placements),
        len(result.model.terms),
    )
    print(
        "Actual objective queries and shots:",
        len(result.evaluations),
        len(result.evaluations) * SHOTS,
    )
    print("Final draws:", sum(count for _, count in result.sampled))
    print("Best sampled mean:", result.selected.mean_cost)
    print("Observed word and cost:", result.word, float(result.model.cost[result.word]))
    print("Observed placements:", result.placements)
    print("Independent minimum cost:", result.exact_cost)
    print("Budget stop and convergence:", result.budget_reached, result.converged)
    print("Native resources:", result.program.report["resources"])
    print(save_trace(result, FOLDER / "trace.png"))
    print(save_costs(result, FOLDER / "costs.png"))
    print(save_packing(result, FOLDER / "packing.png"))


if __name__ == "__main__":
    main()
