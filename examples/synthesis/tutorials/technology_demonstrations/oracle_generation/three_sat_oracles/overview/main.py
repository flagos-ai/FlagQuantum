"""Run and display the complete seeded 3-SAT oracle-generation sweep."""

import argparse
from pathlib import Path

from . import benchmark, plots


def print_report(rows: tuple[benchmark.Row, ...]) -> None:
    print("Problem sizes:", tuple(row.variables for row in rows))
    print("Construction routes:", tuple(row.route for row in rows))
    print("Physical widths:", tuple(row.resources.n_qubits for row in rows))
    print("Static instructions:", tuple(row.resources.n_operations for row in rows))
    print("Local construction seconds:", tuple(row.seconds for row in rows))
    print("Checked reversible instruction traces:", 3 * len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("three-sat-oracle-results")
    )
    args = parser.parse_args()
    rows = benchmark.run()
    print_report(rows)
    plots.save(rows, args.figures / "oracle-generation.png")


if __name__ == "__main__":
    main()
