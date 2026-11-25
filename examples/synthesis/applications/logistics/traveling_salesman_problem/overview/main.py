"""Execute and report the original directed traveling-salesman workflow."""

from pathlib import Path

from .model import visits
from .plots import save_costs, save_graph, save_trace
from .workflow import SHOTS, run_workflow

FOLDER = Path("artifacts/development/traveling-salesman")


def main() -> None:
    result = run_workflow()
    print(
        "Actual queries and objective draws:",
        len(result.evaluations),
        len(result.evaluations) * SHOTS,
    )
    print(
        "Optimized and uniform draws:",
        result.optimized_draw.numel(),
        result.uniform_draw.numel(),
    )
    print("Budget stop and convergence:", result.budget_reached, result.converged)
    print(
        "Observed word and cost:",
        result.best_word,
        float(result.model.cost[result.best_word]),
    )
    print("Observed assignment:", visits(result.best_word))
    print("Observed route:", result.best_route)
    print("Exact route and cost:", result.exact[1], result.exact[0])
    print(save_graph(FOLDER / "graph.png"))
    print(save_trace(result, FOLDER / "trace.png"))
    print(save_costs(result, FOLDER / "costs.png"))
    if result.best_route is None:
        label = "Exact path, observed assignment infeasible"
    else:
        label = f"Observed path, cost {float(result.model.cost[result.best_word]):g}"
    print(
        save_graph(
            FOLDER / "route.png",
            route=result.best_route or result.exact[1],
            title=label,
        )
    )


if __name__ == "__main__":
    main()
