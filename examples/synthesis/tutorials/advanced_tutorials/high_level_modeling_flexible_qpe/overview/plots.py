"""Plot actual finite-shot flexible-QPE estimates and the exact source eigenvalue."""

from pathlib import Path

from matplotlib.figure import Figure

from .workflow import Check


def plot_phase(check: Check, directory: Path) -> Figure:
    if check.comparison:
        raise ValueError("phase plot requires the estimation program")
    figure = Figure(figsize=(9, 5), constrained_layout=True)
    axes = figure.subplots()
    axes.bar(
        [row.word / 32 for row in check.outcomes],
        [row.count for row in check.outcomes],
        width=0.022,
        label="Native samples (1024 configured local draws)",
    )
    axes.axvline(
        check.exact_eigenvalue,
        color="red",
        linestyle="--",
        label="Exact source Hamiltonian eigenvalue",
    )
    axes.set(
        xlabel="Estimated phase",
        ylabel="Observed count",
        title="Five-bit flexible QPE with original per-power Trotter schedule",
        xlim=(0, 1),
    )
    axes.legend()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "phase_estimate.png", dpi=150)
    return figure
