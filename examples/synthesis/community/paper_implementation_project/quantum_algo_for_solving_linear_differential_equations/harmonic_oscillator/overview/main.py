"""Run the complete native time-grid example."""

from pathlib import Path

from .plots import save_comparison
from .workflow import run_workflow, show_observations

OUTPUT = Path("artifacts/development/harmonic-oscillator/solution.png")


def main() -> None:
    """Run the source grid and save the measured comparison."""
    rows = run_workflow()
    show_observations(rows)
    print(save_comparison(rows, OUTPUT))


if __name__ == "__main__":
    main()
