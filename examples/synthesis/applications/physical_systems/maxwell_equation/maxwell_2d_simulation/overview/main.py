"""Execute the complete original-grid Maxwell GQSP simulation."""

import argparse
import json
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print("Original grid=32x32, data amplitudes=4096, width=17, time=5")
    print(
        "Effective time=640, Laurent degree=686, phase rows=1373, walk selections=1372"
    )
    print(
        f"Executed native instructions={result.operations_executed}, resource basis=static_instruction_sequence"
    )
    print(
        f"Actual full-state norm={float(result.state.norm())}, projected norm={result.projection_norm}"
    )
    print(
        f"Absolute projected error={result.absolute_error}, fidelity={result.fidelity}"
    )
    print(f"Independent classical exponential maximum defect={result.classical_defect}")


def write(result: workflow.Result, destination: Path) -> None:
    data = dict(
        grid=32,
        width=17,
        time=5,
        effective_time=640,
        degree=686,
        phase_rows=1373,
        walk_selections=1372,
        operations_executed=result.operations_executed,
        full_state_norm=float(result.state.norm()),
        projection_norm=result.projection_norm,
        absolute_error=result.absolute_error,
        fidelity=result.fidelity,
        classical_defect=result.classical_defect,
        projected=[
            [complex(value).real, complex(value).imag] for value in result.projected
        ],
        classical=[
            [complex(value).real, complex(value).imag] for value in result.classical
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("maxwell-figures"))
    parser.add_argument("--results", type=Path, default=Path("maxwell-results.json"))
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    plots.save(result, args.figures)
    write(result, args.results)


if __name__ == "__main__":
    main()
