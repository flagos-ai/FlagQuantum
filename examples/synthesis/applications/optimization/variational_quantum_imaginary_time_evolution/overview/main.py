"""Execute the complete 14-iteration native MaxCut example."""

from pathlib import Path

from .plots import save_graph, save_trace
from .reference import direct_cut
from .workflow import SHOTS, run_workflow

FOLDER = Path("artifacts/development/imaginary-time-maxcut")


def main() -> None:
    result = run_workflow()
    print("Graph edges:", result.model.edges)
    print(
        "Number of histogram evaluations and shots:",
        result.measurements,
        result.measurements * SHOTS,
    )
    print(
        "Exact positive maximum cut:",
        max(direct_cut(result.model, word) for word in range(16)),
    )
    for row in result.rounds:
        print(
            f"Iteration {row.iteration}:",
            row.energy,
            row.measured_energy,
            row.partition_word,
            row.partition_energy,
        )
    print(save_graph(result.model, FOLDER / "graph.png"))
    print(save_trace(result, FOLDER / "trace.png"))


if __name__ == "__main__":
    main()
