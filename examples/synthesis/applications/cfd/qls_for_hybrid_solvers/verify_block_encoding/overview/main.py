"""Verify both original nozzle matrix block encodings."""

from pathlib import Path

from .plots import draw
from .workflow import run_workflow


def run_example() -> None:
    destination = Path("artifacts/development/cfd-block-encoding")
    for result in run_workflow():
        print(result.method, "width:", result.width, "operations:", result.operations)
        print(
            result.method,
            "scale:",
            result.scale,
            "absolute amplitude error:",
            result.max_error,
        )
        print(draw(result, destination / (result.method + ".png")))


if __name__ == "__main__":
    run_example()
