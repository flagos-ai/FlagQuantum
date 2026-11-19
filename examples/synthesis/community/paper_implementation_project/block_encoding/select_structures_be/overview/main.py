"""Execute all four original-size structured block-encoding comparisons."""

from pathlib import Path

from .plots import save_projections
from .workflow import run_workflow

FOLDER = Path("artifacts/development/structured-blocks")


def main() -> None:
    results = run_workflow()
    for value in results:
        name = value.structure.name
        print(
            f"{name} shape, scale and width:",
            tuple(value.structure.matrix.shape),
            value.scale,
            value.circuit.n_qubits,
        )
        print(f"{name} unnormalized projection:", value.actual.tolist())
        print(f"{name} absolute error:", value.error)
    print(save_projections(results, FOLDER / "projections.png"))


if __name__ == "__main__":
    main()
