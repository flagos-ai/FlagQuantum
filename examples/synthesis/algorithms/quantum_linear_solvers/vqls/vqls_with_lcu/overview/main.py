"""Run the complete source VQLS experiment."""

from pathlib import Path

from .plots import save_comparison, save_trace
from .workflow import BUDGET, SHOTS, run_workflow


def run_example() -> None:
    result = run_workflow()
    last = result.queries[-1]
    print("Actual objective queries:", len(result.queries))
    print("Objective budget and shots per query:", BUDGET, SHOTS)
    print("Total actual objective draws:", len(result.queries) * SHOTS)
    print(
        "Last auxiliary-zero and all-zero counts:", last.auxiliary_zero, last.all_zero
    )
    print("Last sampled cost:", last.cost)
    print("Converged and budget stop:", result.converged, result.budget_stop)
    print("Solution fidelity:", result.fidelity)
    print("Normalized residual:", result.normalized_residual)
    print("Solution amplitudes:", result.amplitudes.real.tolist())
    path = Path("artifacts/development/vqls-lcu")
    print(save_trace(result, path / "trace.png"))
    print(save_comparison(result, path / "solution.png"))


if __name__ == "__main__":
    run_example()
