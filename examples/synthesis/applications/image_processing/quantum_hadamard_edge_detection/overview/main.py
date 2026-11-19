"""Run the full 128-by-128 quantum image edge detection example."""

import argparse
from pathlib import Path

from .plots import histograms, image_panels
from .workflow import SHOTS, Result, run


def print_report(result: Result) -> None:
    print("Image dimensions:", tuple(result.photo.shape))
    print("Shots and observed edge shots:", SHOTS, result.observed_edge_shots)
    print("Exact edge probability:", result.edge_probability)
    print(
        "Preparation and detector maximum errors:",
        result.preparation_error,
        result.state_error,
    )
    print("Workspace probability:", result.work_probability)
    for label, estimate in (
        ("Preparation resources", result.preparation_resources),
        ("Full circuit resources", result.resources),
    ):
        print(label + ":", estimate.n_qubits, estimate.n_operations, estimate.depth)


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image_panels(result, directory / "edge-images.png")
    histograms(result, directory / "edge-histograms.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("edge-detection-figures"))
    options = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, options.figures)


if __name__ == "__main__":
    main()
