"""Report the fresh chemistry, exact reference checks and CX/U3 resources."""

from pathlib import Path

from .model import ACTIVE_ELECTRONS, ACTIVE_SPATIAL, BASIS, GEOMETRY, WIDTH
from .workflow import Result, run_workflow

OUTPUT = Path("artifacts/development/water-exponentiation")


def report(result: Result) -> None:
    print("Geometry in angstroms:", GEOMETRY)
    print("Basis and active space:", BASIS, ACTIVE_SPATIAL, ACTIVE_ELECTRONS)
    print(
        "RHF energy and frozen spatial orbitals:",
        result.model.rhf_energy,
        result.model.frozen_orbitals,
    )
    print("Fresh Pauli width and term count:", WIDTH, len(result.model.terms))
    print(
        "Largest coefficient difference from retained source:",
        result.model.coefficient_difference,
    )
    print("Absolute native Circuit amplitude error:", result.absolute_error)
    print("CX/U3 global-phase-aligned amplitude error:", result.target_error)
    print("CX/U3 circuit resources:", result.target_resources)
    print("Measured probability mass:", sum(p for _, p in result.distribution))


def save_figure(result: Result, output: Path = OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    counts = result.target_resources.operation_counts
    names = ("CX", "U3")
    values = (counts["cx"], counts["u3"])
    fig, axes = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    bars = axes.bar(names, values, color=("#3f708e", "#7a8f65"))
    axes.bar_label(bars, padding=4)
    axes.set(
        title="Water evolution in the CX/U3 basis",
        ylabel="Gate count",
        ylim=(0, max(values) * 1.15),
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "gate-counts.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    result = run_workflow()
    report(result)
    print(save_figure(result))


if __name__ == "__main__":
    main()
