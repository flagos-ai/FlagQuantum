"""Run the original-size two circles, hypercube and bounded CX sweep."""

import argparse
from pathlib import Path

from . import plotting, sweep, walks


def check_endpoints(rows: tuple[sweep.Point, ...]) -> tuple[float, ...]:
    checked: dict[str, float] = {}
    for row in (rows[0], rows[-1]):
        for candidate in (row.arithmetic, row.cascade):
            key = candidate.circuit.content_hash
            if key not in checked:
                checked[key] = sweep.validate(candidate)
    return tuple(checked.values())


def report(
    data: walks.Walks, rows: tuple[sweep.Point, ...], endpoint_errors: tuple[float, ...]
) -> None:
    print(
        "Circle physical widths:",
        (data.circle.resources.n_qubits, data.balanced.resources.n_qubits),
    )
    print("Hypercube physical width:", data.cube.resources.n_qubits)
    print(
        "Native sample totals:",
        tuple(sum(row.counts) for row in (data.circle, data.balanced, data.cube)),
    )
    print(
        "Classical sample totals:",
        (sum(data.classical_circle), sum(data.classical_cube)),
    )
    print(
        "Complete native walk amplitude errors:",
        tuple(row.maximum_error for row in (data.circle, data.balanced, data.cube)),
    )
    print(
        "Clean workspace probabilities:",
        tuple(row.work_probability for row in (data.circle, data.balanced, data.cube)),
    )
    print(
        "Corner observed, exact and classical:",
        (
            data.cube.observed(15),
            data.cube.exact[-1],
            data.classical_cube[-1] / walks.SHOTS,
        ),
    )
    print("Width limits:", tuple(row.width_limit for row in rows))
    print("Arithmetic CX counts:", tuple(sweep.cost(row.arithmetic)[0] for row in rows))
    print("Cascade CX counts:", tuple(sweep.cost(row.cascade)[0] for row in rows))
    print(
        "Selected physical widths:",
        tuple(
            (row.arithmetic.resources.n_qubits, row.cascade.resources.n_qubits)
            for row in rows
        ),
    )
    print("Named-basis endpoint amplitude errors:", endpoint_errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("discrete-walk-results"))
    args = parser.parse_args()
    data = walks.run()
    rows = sweep.sweep()
    report(data, rows, check_endpoints(rows))
    plotting.circle(data.circle, args.figures, classical=data.classical_circle)
    plotting.circle(data.balanced, args.figures)
    plotting.width_sweep(rows, args.figures)


if __name__ == "__main__":
    main()
