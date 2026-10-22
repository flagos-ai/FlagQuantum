"""Render declared sites, commanded paths and the native source schedule.

Install the existing viz extra. Run:
    python -m examples.visualize_rydberg_transport --output transport.svg
Positions come from Core; this view does not infer presence, trap activation,
heating or numerical convergence from a commanded trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from examples.train_rydberg_controls import moving_pair_program
from flagquantum.core.ir.program.physical import RydbergPulse, RydbergTransport


def transport_figure(*, samples: int = 51) -> Figure:
    if type(samples) is not int or samples < 2:
        raise ValueError("display samples must be an integer of at least two")
    program = moving_pair_program()
    record = program.to_ir()
    context = record.physical_context
    assert context is not None
    controls = [op.physical_data for op in record.functions[0].blocks[0].operations]
    end = max(c.start + c.duration for c in controls if c is not None)
    times = torch.linspace(0.0, end, samples, dtype=torch.float64).tolist()
    fig, (geometry, timeline) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    sites = dict(context.sites)
    geometry.scatter(
        [p[0] for p in sites.values()],
        [p[1] for p in sites.values()],
        marker="s",
        s=100,
        facecolors="none",
        edgecolors="#7c8796",
        label="declared sites",
        zorder=2,
    )
    for name, point in sites.items():
        geometry.annotate(
            name,
            point,
            xytext=(0, -19),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for actor, color in zip(program.atom_ids, ("#1767ac", "#da7928"), strict=True):
        positions = [program.commanded_position(actor, t) for t in times]
        x, y = zip(*positions, strict=True)
        geometry.plot(
            x, y, color=color, linewidth=2.5, label=f"{actor} (commanded)", zorder=3
        )
        geometry.scatter([x[0], x[-1]], [y[0], y[-1]], s=35, color=color, zorder=4)
        if positions[-1] != positions[0]:
            geometry.annotate(
                "",
                positions[-1],
                xytext=positions[-2],
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.5},
            )
    geometry.set(
        xlabel=f"x ({context.length_unit})",
        ylabel=f"y ({context.length_unit})",
        title="Declared sites and commanded paths",
        ylim=(-1.0, 1.0),
    )
    geometry.set_aspect("equal", adjustable="box")
    geometry.margins(x=0.15)
    geometry.grid(alpha=0.18)
    geometry.legend(loc="upper left", fontsize=8)
    labels = []
    for row, control in enumerate(controls):
        if isinstance(control, RydbergTransport):
            label = f"move {','.join(control.atoms)} ({control.channel})"
            color = "#da7928"
        else:
            assert isinstance(control, RydbergPulse)
            label = f"pulse {','.join(control.atoms)} ({control.channel})"
            color = "#1767ac"
        labels.append(label)
        timeline.barh(
            row,
            control.duration,
            left=control.start,
            height=0.42,
            color=color,
            alpha=0.85,
        )
    timeline.set_yticks(range(len(labels)), labels=labels)
    timeline.set(
        xlabel=f"time ({context.time_unit})",
        title="Native source schedule",
        xlim=(-0.03 * end, 1.03 * end),
    )
    timeline.grid(axis="x", alpha=0.2)
    timeline.set_axisbelow(True)
    fig.suptitle("Moving Rydberg pair", fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Commanded geometry is not a presence record. Display sampling is not an integration grid.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rydberg_transport.svg"))
    parser.add_argument("--samples", type=int, default=51)
    args = parser.parse_args()
    figure = transport_figure(samples=args.samples)
    try:
        with plt.rc_context({"svg.fonttype": "none"}):
            figure.savefig(args.output, dpi=160)
    finally:
        plt.close(figure)
    program = moving_pair_program()
    print("Commanded b at t=0.5:", program.commanded_position("b", 0.5))
    print("Schedule: move b (aod); pulse a,b (laser)")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
