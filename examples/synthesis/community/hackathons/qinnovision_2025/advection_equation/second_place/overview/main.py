"""Run the complete source-sized second-place advection demonstration."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print(f"Original grid: {len(result.axis)} points, 7 qubits, 32 evolution steps")
    print(
        f"Measured named-basis resources: width={result.width}, depth={result.depth}, CX={result.cx}"
    )
    print(f"Complete absolute state error: {result.state_error:.12g}")
    print(
        f"Classical maximum magnitude: {max(abs(value) for value in result.classical):.12g}"
    )
    print(f"Time sweep: {result.times}")
    print(f"Native CX counts: {result.measured_cx}")
    print(f"Source paper estimates: {result.theory_cx}")
    print(f"Controlled RY widths: {result.cry_widths}")
    print(f"Controlled RY native CX counts: {result.measured_cry}")
    print(f"Controlled RY paper counts: {result.theory_cry}")


def write(result: workflow.Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("advection-figures"))
    parser.add_argument("--results", type=Path, default=Path("advection-results.json"))
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
