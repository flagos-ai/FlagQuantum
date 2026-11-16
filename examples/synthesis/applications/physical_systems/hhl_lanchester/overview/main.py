"""Run the original Lanchester workflow from the command line."""

from pathlib import Path

import torch

from .model import build_model
from .plots import save_matrix_views, save_populations, save_vectors
from .quantum import sampled_swap
from .workflow import show_result, solve

OUTPUT = Path("artifacts/development/lanchester-hhl")


def main() -> None:
    """Regenerate both systems, execute quantum circuits and save comparisons."""
    model = build_model()
    original = torch.linalg.solve(model.operator, model.forcing)
    print(
        save_matrix_views(
            (
                ("Original operator", model.operator),
                ("Forcing vector", model.forcing[:, None]),
            ),
            OUTPUT / "original-system.png",
        )
    )
    print(
        save_vectors((("Direct classical", original),), OUTPUT / "classical-vector.png")
    )
    print(save_populations(original, OUTPUT / "classical-populations.png"))
    print(
        save_matrix_views(
            (
                ("Hermitian embedding", model.hermitian),
                ("Normalized padded vector", model.padded[:, None]),
            ),
            OUTPUT / "hermitian-system.png",
        )
    )
    print(
        save_matrix_views(
            (("Shifted and rescaled operator", model.rescaled),),
            OUTPUT / "shifted-system.png",
        )
    )
    solution = solve(model)
    swap = sampled_swap(solution.logical, solution.reference)
    show_result(solution, swap)
    print(
        save_vectors(
            (
                ("Direct classical", solution.reference),
                ("Native HHL", solution.statevector),
            ),
            OUTPUT / "hhl-vector.png",
        )
    )
    print(
        save_populations(
            original,
            OUTPUT / "hhl-populations.png",
            solution.statevector * model.forcing.norm(),
        )
    )


if __name__ == "__main__":
    main()
