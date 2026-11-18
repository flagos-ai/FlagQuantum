"""Plot actual source locations, evaluated parameters, and sampled routes."""

from pathlib import Path

import matplotlib.pyplot as plt

from . import problem, workflow


def save(result: workflow.Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    positions = result.problem.positions
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for first in range(6):
        for second in range(first + 1, 6):
            ax.plot(
                *zip(positions[first], positions[second], strict=True),
                color="0.85",
                linewidth=0.6,
            )
    for node, (x, y) in enumerate(positions):
        ax.scatter(x, y, s=200, color="lightgreen")
        ax.text(x, y, str(node), ha="center", va="center")
    ax.set(
        title="Six original vehicle-routing locations",
        xlabel="X coordinate",
        ylabel="Y coordinate",
    )
    fig.savefig(directory / "locations.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(range(1, 13), result.parameters[::2], marker="o", label="Gamma")
    ax.plot(range(1, 13), result.parameters[1::2], marker="x", label="Beta")
    ax.set(
        xlabel="QAOA layer",
        ylabel="Angle (radians)",
        title="Best evaluated routing parameters",
    )
    ax.legend()
    fig.savefig(directory / "parameters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    colors = ("tab:blue", "tab:orange")
    for vehicle in range(2):
        route = problem.route(result.selected_word, vehicle)
        for first, second in zip(route, route[1:], strict=False):
            ax.annotate(
                "",
                positions[second],
                positions[first],
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors[vehicle],
                    "lw": 2,
                    "connectionstyle": f"arc3,rad={0.08 if vehicle else -0.08}",
                },
            )
        for node in route:
            x, y = positions[node]
            ax.scatter(x, y, s=220, color=colors[vehicle], zorder=3)
            ax.text(x, y, str(node), ha="center", va="center", color="white", zorder=4)
        ax.plot([], [], color=colors[vehicle], label=f"Vehicle {vehicle}")
    valid = problem.valid(result.selected_word)
    ax.set(
        xlabel="X coordinate",
        ylabel="Y coordinate",
        title=f"Lowest-cost observed assignment • feasible={valid}",
    )
    ax.legend()
    fig.savefig(directory / "routes.png", dpi=140)
    plt.close(fig)
