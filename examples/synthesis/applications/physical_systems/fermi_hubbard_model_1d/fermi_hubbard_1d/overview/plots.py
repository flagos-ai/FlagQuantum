"""Plot measured state amplitudes, charge and spin densities, and limit checks."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from .workflow import Trajectory, one_particle_curves, spread


def save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def amplitudes(trajectory: Trajectory, expected: torch.Tensor, path: Path) -> Path:
    state = trajectory.states[0]
    phase = torch.vdot(expected, state)
    phase /= phase.abs()
    indices = 1 << torch.arange(8)
    qubits = torch.arange(8)
    fig, axes = plt.subplots(figsize=(8, 4))
    axes.bar(qubits - 0.17, expected[indices].real, width=0.34, label="Exact orbital")
    axes.bar(
        qubits + 0.17, (state[indices] / phase).real, width=0.34, label="Native state"
    )
    axes.set(
        xlabel="Mode",
        ylabel="Real amplitude",
        title="Single-electron ground state",
        xticks=range(8),
    )
    axes.legend()
    return save(fig, path)


def initial_densities(trajectory: Trajectory, path: Path) -> Path:
    fig, axes = plt.subplots(figsize=(7, 4))
    sites = range(4)
    axes.plot(sites, trajectory.charge[0], "o-", label="Charge")
    axes.plot(sites, trajectory.spin[0], "s--", label="Spin")
    axes.set(
        xlabel="Site",
        ylabel="Density",
        title="Initial quarter-filling densities",
        xticks=sites,
    )
    axes.legend()
    return save(fig, path)


def dynamics(trajectory: Trajectory, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, i in zip(axes, (0, 4, 9), strict=True):
        ax.plot(range(4), trajectory.charge[i], "o-", label="Charge")
        ax.plot(range(4), trajectory.spin[i], "s--", label="Spin")
        ax.set(
            xlabel="Site",
            title=f"tJ = {float(trajectory.times[i]):.1f}",
            xticks=range(4),
        )
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.suptitle("Interacting model: charge and spin density")
    return save(fig, path)


def spread_plot(trajectory: Trajectory, path: Path) -> Path:
    charge, spin = spread(trajectory)
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(trajectory.times, charge, "o-", label="Charge")
    axes.plot(trajectory.times, spin, "s--", label="Spin")
    axes.set(xlabel="tJ", ylabel="Spread", title="Charge and spin spread")
    axes.legend()
    return save(fig, path)


def free_dynamics(path: Path, trajectory: Trajectory | None = None) -> Path:
    maximum_time = 4.0 if trajectory is None else float(trajectory.times[-1])
    times = torch.linspace(0, maximum_time, 400, dtype=torch.float64)
    _, numerical, analytic = one_particle_curves(times)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for site, ax in enumerate(axes.flat):
        ax.plot(times, numerical[:, site], label="Exact diagonalization")
        if trajectory is None:
            ax.plot(times, analytic[:, site], "--", label="Standing waves")
        else:
            ax.plot(
                trajectory.times,
                trajectory.modes[:, site],
                "o--",
                label="Native Trotter",
            )
        ax.set(xlabel="tJ", ylabel="Occupation", title=f"Site {site}")
        ax.legend(fontsize=8)
    fig.suptitle("One electron on the open chain (U = 0)")
    return save(fig, path)


def frozen_densities(trajectory: Trajectory, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, title, values in zip(
        axes, ("Charge", "Spin"), (trajectory.charge, trajectory.spin), strict=True
    ):
        ax.plot(range(4), values[-1], "o", label="Native at t = 3")
        ax.plot(range(4), values[0], "x", ms=9, label="Exact initial value")
        ax.set(xlabel="Site", ylabel="Density", title=title, xticks=range(4))
        ax.legend()
    fig.suptitle("Vanishing hopping (J = 0): conserved densities")
    return save(fig, path)
