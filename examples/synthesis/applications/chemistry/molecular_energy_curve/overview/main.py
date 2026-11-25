"""Plot the full distance sweep and report energies of executed variational states."""

from pathlib import Path

from .workflow import ITERATION_LIMIT, Point, run_workflow

OUTPUT = Path("artifacts/development/hydrogen-energy-curve")


def report(points: tuple[Point, ...]) -> None:
    print("Native optimizer iteration limit:", ITERATION_LIMIT)
    print("Distance grid:", tuple(row.distance for row in points))
    print("RHF energies:", tuple(row.model.rhf for row in points))
    print("FCI energies:", tuple(row.model.fci for row in points))
    print("Native VQE energies:", tuple(row.energy for row in points))
    print(
        "Objective evaluations by distance:", tuple(row.evaluations for row in points)
    )
    print("Convergence by distance:", tuple(row.converged for row in points))
    print(
        "Max VQE-to-FCI energy difference:",
        max(abs(row.energy - row.model.fci) for row in points),
    )
    print(
        "Max paired-to-complement coupling:",
        max(row.model.coupling_to_complement for row in points),
    )
    print("Physical register widths:", tuple(row.resources.n_qubits for row in points))
    print("Local elapsed seconds by distance:", tuple(row.seconds for row in points))


def save_figure(points: tuple[Point, ...], output: Path = OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    positions = [row.distance for row in points]
    figure, axis = plt.subplots(figsize=(8.25, 4.5), layout="constrained")
    axis.plot(
        positions,
        [row.energy for row in points],
        "--",
        color="#ae4836",
        label="FlagQuantum VQE",
    )
    axis.plot(
        positions,
        [row.model.rhf for row in points],
        "s",
        color="#366f91",
        label="Hartree–Fock",
    )
    axis.plot(
        positions,
        [row.model.fci for row in points],
        "^",
        markerfacecolor="none",
        color="#657e3a",
        label="Full configuration interaction",
    )
    axis.set(
        title="Hydrogen binding curve in STO-3G",
        xlabel="Distance (Å)",
        ylabel="Energy (hartree)",
    )
    axis.legend()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "binding-curve.png"
    figure.savefig(path, dpi=135)
    plt.close(figure)
    return path


def main() -> None:
    points = run_workflow()
    report(points)
    save_figure(points)


if __name__ == "__main__":
    main()
