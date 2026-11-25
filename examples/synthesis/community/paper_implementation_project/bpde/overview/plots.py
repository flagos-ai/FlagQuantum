"""Render the source's six energy, time and posterior comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt

from . import model, workflow

BLUE = "#2b6a9e"
ORANGE = "#c66b22"


def save(result: workflow.Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.7, 4.1))
    try:
        axis.plot(
            tuple(row.distance for row in result.curve),
            tuple(row.fci for row in result.curve),
            "o-",
            color=BLUE,
            markersize=3,
        )
        axis.set_xlabel("Internuclear distance (Å)")
        axis.set_ylabel("Ground state energy (Hartree)")
        axis.set_title("Fresh hydrogen ground-state curve")
        figure.tight_layout()
        figure.savefig(directory / "energy-curve.png", dpi=140)
    finally:
        plt.close(figure)

    spacing, values = model.differences(result.curve)
    trial = max(
        range(len(result.diagnostic.energies)),
        key=result.diagnostic.z_estimates.__getitem__,
    )
    guess = result.diagnostic.energies[trial] / model.DISPLACEMENT
    posterior = result.history[-1]
    for name, comparison in (
        ("finite-difference", None),
        ("grid-maximum", (guess, 0.0, "Largest observed grid signal")),
        (
            "posterior",
            (
                posterior.posterior_mu / model.DISPLACEMENT,
                posterior.posterior_sigma / model.DISPLACEMENT,
                "Bayesian estimate",
            ),
        ),
    ):
        figure, axis = plt.subplots(figsize=(6.7, 4.1))
        try:
            axis.plot(
                spacing,
                values,
                "o-",
                color=BLUE,
                markersize=3,
                label="Fresh FCI difference",
            )
            axis.axhline(0, color="#58616b", linestyle="--", linewidth=0.8)
            if comparison is not None:
                estimate, error, label = comparison
                axis.errorbar(
                    model.CENTRE,
                    estimate,
                    yerr=error,
                    fmt="x",
                    capsize=4,
                    color=ORANGE,
                    label=label,
                )
                axis.legend(loc="lower right")
            axis.set_xlabel("Internuclear distance (Å)")
            axis.set_ylabel("Energy gradient (Hartree/Å)")
            axis.set_title("Hydrogen energy gradient")
            axis.set_xlim(0.45, 2.05)
            axis.set_ylim(-1, 0.5)
            figure.tight_layout()
            figure.savefig(directory / f"{name}.png", dpi=140)
        finally:
            plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.7, 4.1))
    try:
        middle = result.diagnostic
        axis.plot(middle.energies, middle.z_estimates, "o-", color=BLUE)
        axis.set_xlabel("Trial difference (Hartree)")
        axis.set_ylabel("Observed readout Z")
        axis.set_title(f"Five-offset scan at time {middle.time:g}")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        figure.tight_layout()
        figure.savefig(directory / "scan.png", dpi=140)
    finally:
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    try:
        for axis, scan in zip(axes, result.study, strict=True):
            axis.plot(scan.energies, scan.z_estimates, "o-", color=BLUE)
            axis.set_title(f"Time {scan.time:g}")
            axis.set_ylim(0.9900, 1.0001)
            axis.set_xlabel("Trial difference")
            axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[0].set_ylabel("Observed readout Z")
        figure.tight_layout()
        figure.savefig(directory / "time-study.png", dpi=140)
    finally:
        plt.close(figure)
