"""Run the complete original minimum dominating set example."""

from pathlib import Path

from .plots import draw_results
from .workflow import SHOTS, run_workflow


def run_example() -> None:
    result = run_workflow()
    print("Closed neighborhoods:", result.model.neighborhoods)
    print("Slack weights:", result.model.slack_weights)
    print("Selected circuit resources:", result.program.report["resources"])
    print("Actual objective evaluations:", len(result.trace))
    print(
        "Initial and selected CVaR:",
        result.trace[0].cost,
        min(r.cost for r in result.trace),
    )
    print("Optimizer converged:", result.optimizer.converged)
    print("Shots per comparison:", SHOTS)
    print("Five lowest observed penalized costs:", result.optimized[:5])
    print("Exact decision minima:", result.classical)
    print(
        "Figures:",
        draw_results(result, Path("artifacts/development/minimum-dominating-set")),
    )


if __name__ == "__main__":
    run_example()
