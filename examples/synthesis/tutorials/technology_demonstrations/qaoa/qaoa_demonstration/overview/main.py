"""Compare newly sampled FlagQuantum energies with the original reference trace."""

import json
from pathlib import Path

from .model import SHOTS
from .workflow import MAX_ITERATIONS, MAX_QUERIES, Run, run_workflow

OUTPUT = Path("artifacts/development/qaoa-clique-benchmark")


def historical_energies() -> tuple[float, ...]:
    source = Path(__file__).with_name("qiskit_reference.json")
    data = json.loads(source.read_text())
    values = tuple(float(value) for value in data["data"])
    if len(values) != data["rows"]:
        raise ValueError("The historical trace does not match its provenance")
    return values


def save_figure(run: Run, output: Path = OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    reference = historical_energies()
    native = tuple(row.cost for row in run.trace)
    axes.plot(
        range(1, len(reference) + 1),
        reference,
        label="Source Qiskit reference (depth 1,646, 1,660 CX)",
        color="#5575a7",
        linewidth=1.25,
    )
    axes.plot(
        range(1, len(native) + 1),
        native,
        label=(
            f"FlagQuantum (depth {run.resources.depth:,}, "
            f"{run.resources.operation_counts.get('cx', 0):,} CX)"
        ),
        color="#b84931",
        linewidth=1.25,
    )
    axes.set(
        title="Ten-layer QAOA for the seven-vertex clique model",
        xlabel="Trace index (native measured objective query)",
        ylabel="Energy",
    )
    axes.legend()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "energy-traces.png"
    figure.savefig(path, dpi=145)
    plt.close(figure)
    return path


def report(run: Run) -> None:
    print(
        "Graph edges and conflicts:",
        len(run.model.edges),
        len(run.model.conflicts),
    )
    print("Optimizer budgets:", MAX_ITERATIONS, MAX_QUERIES)
    print(
        "Native iterations, queries, converged, query cap:",
        run.iterations,
        len(run.trace),
        run.converged,
        run.budget_reached,
    )
    print("Native objective shots and final shots:", SHOTS * len(run.trace), SHOTS)
    print("Lowest observed mean energy:", run.mean)
    print(
        "Selected observed nodes, shots and feasible:", run.nodes, run.count, run.clique
    )
    print("FlagQuantum CX/U3 converted resources:", run.resources)
    print("Native optimizer and resource-profile elapsed seconds:", run.seconds)
    print("Historical reference trace length:", len(historical_energies()))


def main() -> None:
    run = run_workflow()
    report(run)
    print(save_figure(run))


if __name__ == "__main__":
    main()
