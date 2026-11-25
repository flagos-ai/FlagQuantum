"""Run every original preparation and report finite samples."""

from pathlib import Path

from .plots import save_distributions
from .workflow import SCENARIOS, SHOTS, run_case

OUTPUT = Path("artifacts/development/symmetric-compression/distributions.png")


def main() -> None:
    results = tuple(
        run_case(scenario, 2026 + index) for index, scenario in enumerate(SCENARIOS)
    )
    print("Shots per scenario:", SHOTS)
    for result in results:
        print(f"{result.scenario} counts:", result.counts)
        print(f"{result.scenario} probabilities:", result.probabilities)
        print(f"{result.scenario} resources:", result.program.report["resources"])
    print(save_distributions(results, OUTPUT))


if __name__ == "__main__":
    main()
