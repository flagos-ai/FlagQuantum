"""Execute all original native Max-XORSAT stages."""

from pathlib import Path

from .model import majority_errors
from .plots import save_graph, save_scores
from .workflow import run_workflow

FOLDER = Path("artifacts/development/dqi-workflow")


def main() -> None:
    result = run_workflow()
    print("Unary observation:", result.unary)
    print("Dicke support sizes:", len(result.dicke_four), len(result.dicke_two))
    print(
        "Decoder entries and majority misses:",
        len(result.model.syndrome_table),
        majority_errors(result.model),
    )
    print("Optimal weight amplitudes:", result.model.weights)
    print("DQI shots:", sum(count for _, count in result.dqi))
    print("Sampled DQI score:", result.mean_score)
    print(
        "Exact uniform score:",
        sum(result.model.objective) / len(result.model.objective),
    )
    print("Native resources:", result.program.report["resources"])
    print(save_graph(FOLDER / "graph.png"))
    print(save_scores(result, FOLDER / "scores.png"))


if __name__ == "__main__":
    main()
