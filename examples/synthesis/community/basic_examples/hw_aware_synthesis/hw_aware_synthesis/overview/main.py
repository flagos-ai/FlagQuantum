"""Run the local hardware-aware MCX comparison."""

import argparse
import math
from pathlib import Path

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import Program

from .workflow import (
    GRID_EDGES,
    GRID_WIDTH,
    STAR_EDGES,
    Compilation,
    Topology,
    build_program,
    route,
    verify,
)


def report_baseline(program: Program) -> None:
    cost = program.report["resources"]
    assert isinstance(cost, ResourceEstimate)
    print("10-control baseline:", cost.n_qubits, cost.n_operations, cost.depth)
    observed = dict(program.probabilities(fq.run(program), registers=("target",)))
    assert abs(observed[(0,)] - 1) < 1e-12
    print("10-control target at zero:", observed)
    print(
        "10-control state check:",
        verify(
            program.circuit,
            (*program.registers["controls"], *program.registers["target"]),
        ),
    )


def report_routing(comparison: Compilation) -> None:
    for candidate in comparison.candidates:
        print(
            f"{comparison.topology} candidate:",
            candidate.label,
            candidate.resources.operation_counts["cx"],
            candidate.resources.depth,
        )
    row = comparison.selected
    print(
        f"{comparison.topology} selected:",
        row.label,
        row.resources.n_qubits,
        row.resources.operation_counts["cx"],
        row.resources.depth,
    )
    print(f"{comparison.topology} state check:", verify(row.circuit, row.logical_wires))


def plot_topology(topology: Topology, directory: Path) -> None:
    import matplotlib.pyplot as plt

    coordinates: list[tuple[float, float]]
    if topology == "grid":
        coordinates = [(i % 6, -(i // 6)) for i in range(GRID_WIDTH)]
        edges = GRID_EDGES
        title = "3 × 6 grid connectivity"
    else:
        angles = (math.tau * i / (GRID_WIDTH - 1) for i in range(GRID_WIDTH - 1))
        coordinates = [(0.0, 0.0)] + [
            (math.cos(angle), math.sin(angle)) for angle in angles
        ]
        edges = STAR_EDGES
        title = "18-qubit star connectivity"
    fig, axes = plt.subplots(figsize=(7, 5), layout="constrained")
    for first, second in edges:
        start, end = coordinates[first], coordinates[second]
        axes.plot(
            (start[0], end[0]), (start[1], end[1]), color="#8098ae", linewidth=1.6
        )
    x, y = zip(*coordinates, strict=True)
    axes.scatter(x, y, s=550, color="#225980", zorder=3)
    for i, (px, py) in enumerate(coordinates):
        axes.text(px, py, str(i), color="white", ha="center", va="center", zorder=4)
    axes.set(aspect="equal", title=title)
    axes.margins(0.15)
    axes.set_axis_off()
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{topology}-connectivity.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("mcx-figures"))
    options = parser.parse_args()
    report_baseline(build_program(10))
    program = build_program(12)
    topologies: tuple[Topology, ...] = ("grid", "star")
    for topology in topologies:
        plot_topology(topology, options.figures)
        report_routing(route(program, topology))


if __name__ == "__main__":
    main()
