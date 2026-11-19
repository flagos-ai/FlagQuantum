"""Run the original seven-data-bit Circuit block-projection experiment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import plots, workflow


def report(result: workflow.Result) -> None:
    print("Original grid: 8 spatial by 8 velocity points, two E sectors")
    print(
        f"Block encoding width={result.block_resources.n_qubits}, operations={result.block_resources.n_operations}, depth={result.block_resources.depth}"
    )
    print(
        f"Source demonstration QSVT pair width={result.qsvt_resources.n_qubits}, operations={result.qsvt_resources.n_operations}, depth={result.qsvt_resources.depth}"
    )
    print(
        f"Resource basis={result.block_resources.basis}, largest matrix support={result.block_resources.max_operation_width}"
    )
    print(
        f"Complete complex projected-vector L2 error={result.l2_error}, auxiliary leak={result.work_leak}"
    )


def write(result: workflow.Result, path: Path) -> None:
    data = asdict(result)
    data["expected"] = [[value.real, value.imag] for value in result.expected]
    data["projected"] = [[value.real, value.imag] for value in result.projected]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("vlasov-circuit-figures"))
    parser.add_argument(
        "--results", type=Path, default=Path("vlasov-circuit-results.json")
    )
    args = parser.parse_args()
    result = workflow.run()
    report(result)
    write(result, args.results)
    plots.save(result, args.figures)


if __name__ == "__main__":
    main()
