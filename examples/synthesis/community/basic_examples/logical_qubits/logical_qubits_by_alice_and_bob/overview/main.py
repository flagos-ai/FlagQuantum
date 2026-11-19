"""Execute the complete original logical-cat parameter study locally."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print(
        "Original input: RandomState12, two identical three-bit states, seven physical bits"
    )
    print(f"RX(0.1) Clifford+T examples: {result.rotation_examples}")
    print(f"Noiseless actual10000-draw counts: {result.noiseless.counts}")
    print(
        f"Configured distance31/photons19/kappa2=1e4 counts: {result.configured.counts}"
    )
    print(
        f"Original grid: {len(result.grid)} points, actual draws={len(result.grid)*workflow.SHOTS}"
    )
    print(f"Total actual draws={(len(result.grid)+2)*workflow.SHOTS}")
    print(f"Complete exact preparation error={result.full_state_preparation_error}")
    print(f"Largest compiled RY operator error={result.maximum_rotation_error}")
    print(
        "Resource basis: physical Clifford+T instructions and exact ASAP correction clocks; merged CPTP nodes reported separately"
    )


def write(result: workflow.Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("logical-cat-figures"))
    parser.add_argument(
        "--results", type=Path, default=Path("logical-cat-results.json")
    )
    arguments = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, arguments.figures)
    write(result, arguments.results)


if __name__ == "__main__":
    main()
