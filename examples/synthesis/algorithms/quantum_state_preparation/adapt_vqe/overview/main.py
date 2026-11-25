"""Run both adaptive pathways and the fixed VQE comparison locally."""

from pathlib import Path

from .model import OUTER_BUDGET, TOLERANCE, build_model
from .plots import save_trace
from .workflow import AdaptiveRun, Optimization, adapt, fixed

OUTPUT = Path("artifacts/development/adapt-vqe")


def show_run(label: str, run: AdaptiveRun, ground_energy: float) -> None:
    """Describe the actual selected operators, energies and stop condition."""
    for index, row in enumerate(run.steps, start=1):
        print(
            f"{label} step {index}: chosen={row.chosen_index}, "
            f"active={row.fit.indices}, energy={row.fit.energy:.12f}, "
            f"gradient norm={row.gradient_norm:.12f}, evaluations={row.fit.evaluations}"
        )
    best = run.best.fit
    print(f"{label} stop: {run.stop}")
    print(f"{label} optimized angles: {best.angles}")
    print(f"{label} best energy: {best.energy:.12f}")
    print(
        f"{label} relative error (%): {100 * abs((best.energy - ground_energy) / ground_energy):.9f}"
    )
    print(
        f"{label} best width and depth: {best.resources.n_qubits}, {best.resources.depth}"
    )


def save_run(label: str, run: AdaptiveRun, ground_energy: float) -> tuple[Path, Path]:
    """Render the source's energy and gradient figures for one adaptive loop."""
    energy = save_trace(
        [row.fit.energy for row in run.steps],
        ground_energy,
        title=f"{label} ADAPT-VQE energy",
        quantity="Energy",
        reference_label="Exact ground energy",
        path=OUTPUT / f"{label.lower()}-energy.png",
    )
    gradient = save_trace(
        [row.gradient_norm for row in run.steps],
        TOLERANCE,
        title=f"{label} ADAPT-VQE gradients",
        quantity="Scaled gradient norm",
        reference_label="Source tolerance",
        path=OUTPUT / f"{label.lower()}-gradient.png",
    )
    return energy, gradient


def show_fixed(label: str, fit: Optimization, ground_energy: float) -> None:
    """Report the actual fixed schedule and the final resource estimate."""
    print(f"{label} schedule: {fit.indices}")
    print(f"{label} energy: {fit.energy:.12f}")
    print(f"{label} optimized angles: {fit.angles}")
    print(
        f"{label} relative error (%): {100 * abs((fit.energy - ground_energy) / ground_energy):.9f}"
    )
    print(f"{label} width and depth: {fit.resources.n_qubits}, {fit.resources.depth}")


def main() -> None:
    """Run both source adaptive schedules and the two fixed interpretations."""
    model = build_model()
    print(f"Exact ground energy: {model.ground_energy:.12f}")
    standard = adapt(model, modified=False)
    show_run("Standard", standard, model.ground_energy)
    save_run("Standard", standard, model.ground_energy)
    modified = adapt(model, modified=True)
    show_run("Modified", modified, model.ground_energy)
    save_run("Modified", modified, model.ground_energy)
    show_fixed("Modular fixed", fixed(model), model.ground_energy)
    if len(modified.selections) == OUTER_BUDGET:
        show_fixed(
            "Captured fixed", fixed(model, modified.selections), model.ground_energy
        )


if __name__ == "__main__":
    main()
