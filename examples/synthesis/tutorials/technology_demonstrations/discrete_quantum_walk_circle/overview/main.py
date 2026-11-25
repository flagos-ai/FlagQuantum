"""Run seven original circle sizes and regenerate the four comparison plots."""

import argparse
from pathlib import Path

from . import archive, benchmark, plots


def verify(rows: tuple[benchmark.Entry, ...]) -> tuple[float, ...]:
    return tuple(
        benchmark.check(candidate, row.position_bits)
        for row in (rows[0], rows[-1])
        for candidate in (row.width_first, row.cx_first)
    )


def report(rows: tuple[benchmark.Entry, ...], state_errors: tuple[float, ...]) -> None:
    print("Position bits:", tuple(row.position_bits for row in rows))
    print(
        "Width-first wires:", tuple(row.width_first.resources.n_qubits for row in rows)
    )
    print("Width-first CX:", tuple(row.width_first.cx for row in rows))
    print("CX-first wires:", tuple(row.cx_first.resources.n_qubits for row in rows))
    print("CX-first CX:", tuple(row.cx_first.cx for row in rows))
    print("Visited candidates:", tuple(row.candidate_count for row in rows))
    print("Width-first local seconds:", tuple(row.width_seconds for row in rows))
    print("CX-first local seconds:", tuple(row.cx_seconds for row in rows))
    print("Complete endpoint state errors:", state_errors)
    print("Published CSV rows per method:", tuple(map(len, archive.read())))
    print("Published Qiskit rows:", len(archive.QISKIT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("circle-generation-results")
    )
    args = parser.parse_args()
    rows = benchmark.run()
    report(rows, verify(rows))
    current = benchmark.records(rows)
    historical = archive.read()
    for name, data, archived in (
        ("local", current, False),
        ("published", historical, True),
    ):
        plots.save(data, "cx", args.figures / f"{name}-cx.png", historical=archived)
        plots.save(
            data, "seconds", args.figures / f"{name}-seconds.png", historical=archived
        )


if __name__ == "__main__":
    main()
