"""Read-only incidence views of existing native tensor-network nodes.

The locked Qualtran tensor notebook calls Quimb's network inspection/drawing.
These native presentations consume existing nodes and register-index side tables;
they do not implement Quimb layouts or create an execution graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..simulation.tensor_network.models import TensorNetworkNode
from ..simulation.tensor_network.path_search import _label_dims

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def tensor_index_occurrences(
    nodes: Sequence[TensorNetworkNode],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """List (tensor ordinal, axis) incidences in first-seen integer-label order.

    A single occurrence is an outer index. Multiple occurrences are a summed
    index, including repeated axes of one tensor and hyperedges shared by more
    than two tensors. Shape consistency uses Simulation's existing validator.
    No tensor values, contraction plan or device transfer are requested.
    Returned dictionaries are independent of the input; node data stays shared.
    """
    _label_dims(nodes)
    incidences: dict[int, list[tuple[int, int]]] = {}
    for ordinal, node in enumerate(nodes):
        for axis, label in enumerate(node.labels):
            if type(label) is not int or label < 0:
                raise ValueError("tensor drawing labels must be nonnegative integers")
            incidences.setdefault(label, []).append((ordinal, axis))
    return {label: tuple(axes) for label, axes in incidences.items()}


def _display_names(
    occurrences: Mapping[int, tuple[tuple[int, int], ...]],
    names: Mapping[int, str] | None,
) -> dict[int, str]:
    if names is not None and any(label not in occurrences for label in names):
        raise ValueError("tensor display names must refer to existing indices")
    return {
        label: names[label] if names is not None and label in names else str(label)
        for label in occurrences
    }


def export_tensor_network_dot(
    nodes: Sequence[TensorNetworkNode],
    *,
    index_names: Mapping[int, str] | None = None,
) -> str:
    """Export every tensor axis as an incidence edge, without running Graphviz.

    Tensor and index ordinals provide identities; display names may collide.
    Scalars remain isolated tensor vertices. Separate index vertices preserve
    hyperedges and repeated trace axes rather than silently turning them into
    ordinary two-endpoint edges. This is topology parity, not Quimb layout parity.
    """
    occurrences = tensor_index_occurrences(nodes)
    names = _display_names(occurrences, index_names)
    lines = ["graph tensors {", "  rankdir=LR;"]
    for ordinal, node in enumerate(nodes):
        name = node.name or f"T{ordinal}"
        lines.append(f"  t{ordinal} [shape=ellipse, label={json.dumps(name)}];")
    for ordinal, (label, axes) in enumerate(occurrences.items()):
        tensor, axis = axes[0]
        dimension = int(nodes[tensor].tensor.shape[axis])
        caption = json.dumps(f"{names[label]} [dim={dimension}]")
        lines.append(f"  i{ordinal} [shape=diamond, label={caption}];")
        for tensor, axis in axes:
            lines.append(f'  t{tensor} -- i{ordinal} [label="axis {axis}"];')
    return "\n".join((*lines, "}"))


def draw_tensor_network(
    nodes: Sequence[TensorNetworkNode],
    *,
    index_names: Mapping[int, str] | None = None,
    node_colors: Mapping[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Draw a deterministic bipartite tensor/index view; return caller-owned figure.

    Nodes use their existing names, with optional exact-name colors. Integer
    execution labels are never renamed, even when friendly display names alias.
    Each axis gets a line and axis number, retaining repeated incidences. Only
    shape and label metadata are read; even meta-device tensors can be drawn.
    The finite diagnostic layout does not promise Quimb spectral coordinates,
    optimizer information, or readability for arbitrarily large networks.
    """
    from matplotlib import pyplot as plt

    occurrences = tensor_index_occurrences(nodes)
    names = _display_names(occurrences, index_names)
    rows = max(len(nodes), len(occurrences), 1)
    figure, axes = plt.subplots(figsize=(8, min(16, max(3, 0.6 * rows))))
    positions = {
        ordinal: (len(nodes) - 1) / 2 - ordinal for ordinal in range(len(nodes))
    }
    for ordinal, (label, endpoints) in enumerate(occurrences.items()):
        y = (len(occurrences) - 1) / 2 - ordinal
        axes.scatter([2], [y], marker="D", s=45, color="#56647a", zorder=3)
        tensor, axis = endpoints[0]
        dimension = int(nodes[tensor].tensor.shape[axis])
        axes.text(2.12, y, f"{names[label]} [{dimension}]", va="center", fontsize=9)
        for tensor, axis in endpoints:
            start_y = positions[tensor] + 0.06 * (
                axis - (len(nodes[tensor].labels) - 1) / 2
            )
            axes.plot([0.12, 1.96], [start_y, y], color="#9aa5b4", linewidth=1)
            axes.text(
                1.04,
                (start_y + y) / 2,
                str(axis),
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
    for ordinal, node in enumerate(nodes):
        color = (
            node_colors.get(node.name, "#d9e9ff")
            if node_colors is not None
            else "#d9e9ff"
        )
        axes.scatter(
            [0],
            [positions[ordinal]],
            s=500,
            color=color,
            edgecolors="#34445c",
            zorder=4,
        )
        axes.text(
            -0.17,
            positions[ordinal],
            node.name or f"T{ordinal}",
            ha="right",
            va="center",
        )
    if not nodes:
        axes.text(1, 0, "Empty tensor network", ha="center")
    axes.set_xlim(-0.8, 3.3)
    axes.set_ylim(-rows / 2 - 0.5, rows / 2 + 0.5)
    axes.set_axis_off()
    axes.set_title("Tensor nodes and index incidences")
    figure.tight_layout()
    return figure, axes
