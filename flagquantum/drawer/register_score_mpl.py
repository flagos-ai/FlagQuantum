# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 musical-score Matplotlib rendering.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Render register musical-score presentation records with Matplotlib."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .register_score_data import MusicalScoreData, ScorePort, ScoreVerticalLine

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _draw_text(ax: Axes, x: float, y: float, text: str, fontsize: int) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transData,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox={"lw": 0, "fc": "white"},
    )


def draw_register_score_port(ax: Axes, port: ScorePort) -> None:
    """Draw one builtin glyph using the record's sequential x and positive row y."""
    x, y = port["seq_x"], -port["y"]
    symbol = port["symb_cls"]
    attributes = port["symb_attributes"]
    if symbol == "Circle":
        from matplotlib.patches import Circle

        color = "k" if attributes["filled"] else "w"
        ax.add_patch(Circle((x, y), radius=0.1, fc=color, ec="k"))
        return
    if symbol == "Text":
        _draw_text(ax, x, y, str(attributes["text"]), int(attributes["fontsize"]))
        return
    if symbol == "ModPlus":
        text, fontsize = "⊕", 18
        bbox: dict[str, str | int] = {"fc": "none", "lw": 0}
    else:
        text, fontsize = str(attributes["text"]), 10
        style = {
            "TextBox": "round",
            "RarrowTextBox": "rarrow",
            "LarrowTextBox": "larrow",
        }
        bbox = {"boxstyle": style[symbol], "fc": "white"}
    ax.text(
        x,
        y,
        text,
        transform=ax.transData,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=bbox,
    )


def draw_register_score_label(ax: Axes, connector: ScoreVerticalLine) -> None:
    """Draw a connector's Text label at source row bottom_y minus one half."""
    label = connector["label"]
    _draw_text(
        ax,
        connector["x"],
        0.5 - connector["bottom_y"],
        label["text"],
        label["fontsize"],
    )


def draw_register_score(
    data: MusicalScoreData,
    *,
    unit_to_inches: float = 0.8,
    max_width: float = 10.0,
    max_height: float = 8.0,
    port_drawer: Callable[[Axes, ScorePort], None] | None = None,
    label_drawer: Callable[[Axes, ScoreVerticalLine], None] | None = None,
) -> tuple[Figure, Axes]:
    """Draw the sequential score and return the caller-owned figure.

    port_drawer replaces drawing for every port, including boundary labels, in
    record order. It may delegate builtin glyphs to draw_register_score_port.
    Coordinates retain the record's positive row convention. The callback owns
    its custom presentation and treats the supplied record as read-only.
    label_drawer replaces each nonempty connector label after its line is
    drawn, before all port calls. It receives the existing connector record
    and may delegate to draw_register_score_label. Its source row coordinate
    is bottom_y minus one half, negated for Matplotlib.
    Callback failures close the incomplete figure and propagate unchanged.
    """
    from matplotlib import pyplot as plt

    width = unit_to_inches * (data["max_x"] + 3.0)
    height = unit_to_inches * (data["max_y"] + 1.5)
    if width > height and width > max_width:
        scale = max_width / width
        width *= scale
        height *= scale
    elif height > max_height:
        scale = max_height / height
        height *= scale
        width *= scale

    fig, ax = plt.subplots(figsize=(width, height))
    for line in data["hlines"]:
        color = "b" if line["flavor"] == "HLineFlavor.CLASSICAL" else "k"
        ax.hlines(
            -line["y"], line["seq_x_start"], line["seq_x_end"], color=color, zorder=-1
        )
    draw_port = draw_register_score_port if port_drawer is None else port_drawer
    draw_label = draw_register_score_label if label_drawer is None else label_drawer
    try:
        for connector in data["vlines"]:
            x = connector["x"]
            bottom = connector["bottom_y"]
            ax.vlines(x, -connector["top_y"], -bottom, color="k", zorder=-1)
            if connector["label"]["text"]:
                draw_label(ax, connector)
        for port in data["soqs"]:
            draw_port(ax, port)
    except BaseException:
        plt.close(fig)
        raise

    ax.set_xlim(-2, data["max_x"] + 1)
    ax.set_ylim(-data["max_y"] - 0.5, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax
