"""Headless coverage for the Matplotlib renderer's per-gate branches.

The drawer owns layout, symbols, and presentation options; these tests pin the
rendered artefacts (text, patches, axes limits) for each gate family so a
renderer change cannot silently drop a gate from the figure. Rendering uses the
`Agg` backend, as `flagquantum/drawer/README.md` requires.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit

# mpl_drawer imports Matplotlib at module scope, so the whole module is skipped
# from collection when the optional `viz` extra is absent rather than failing
# the torch-only lanes.
if importlib.util.find_spec("matplotlib") is None:  # pragma: no cover
    pytest.skip(
        "matplotlib is an optional dependency (flagquantum[viz])",
        allow_module_level=True,
    )

from flagquantum.drawer.mpl_drawer import MPLDrawer, draw_mpl  # noqa: E402


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _texts(ax: Any) -> list[str]:
    return [text.get_text() for text in ax.texts]


class _LegacyDevice:
    """A device-shaped object exposing only an operation history."""

    def __init__(
        self, op_history: list[dict[str, Any]], n_wires: int | None = 3
    ) -> None:
        self.op_history = op_history
        if n_wires is not None:
            self.n_wires = n_wires


def _render(program: object, **kwargs: Any) -> tuple[Any, Any]:
    fig, ax = draw_mpl(program, **kwargs)
    return fig, ax


def test_single_qubit_gates_render_one_box_per_gate() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(3)
    circuit.h(0).x(1).y(2).s(0).t(1).z(2)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == ["H", "X", "Y", "S", "T", "Z"]
        assert len(ax.patches) == 6
    finally:
        plt.close(fig)


def test_parameterized_gates_render_symbol_and_formatted_angle() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.25).crx(0, 1, 0.5)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == ["RY\n0.25", "CRX\n0.5"]
    finally:
        plt.close(fig)


def test_decimals_none_hides_gate_parameters() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(1)
    circuit.ry(0, theta=0.25)

    fig, ax = _render(circuit, decimals=None, show_wire_labels=False)
    try:
        assert _texts(ax) == ["RY"]
    finally:
        plt.close(fig)


def test_parameterized_and_phase_gates_use_their_own_colours() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.25).p(1, 0.5)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        colours = [patch.get_facecolor() for patch in ax.patches]
        assert len(colours) == 2
        assert colours[0] != colours[1]
    finally:
        plt.close(fig)


def test_controlled_gates_draw_a_control_point_and_marker() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.cz(0, 1)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == ["Z"]
        assert len(ax.patches) == 2
    finally:
        plt.close(fig)


def test_cx_renders_without_a_lettered_target_box() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.cx(0, 1)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == []
        assert len(ax.patches) == 2
    finally:
        plt.close(fig)


def test_controlled_phase_gate_renders_its_parameter() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.cphase(0, 1, 0.5)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == ["CP\n0.5"]
    finally:
        plt.close(fig)


def test_ising_gates_render_a_label_spanning_their_wires() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(3)
    circuit.rxx(0, 1, 0.3).ryy(1, 2, 0.3).rzz(0, 2, 0.3)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert [text.split("\n")[0] for text in _texts(ax)] == ["RXX", "RYY", "RZZ"]
        assert len(ax.patches) == 3
    finally:
        plt.close(fig)


def test_toffoli_and_cswap_render_three_wire_gates() -> None:
    plt = _pyplot()
    toffoli = fq.Circuit(3)
    toffoli.ccx(0, 1, 2)
    cswap = fq.Circuit(3)
    cswap.cswap(0, 1, 2)

    fig, ax = _render(toffoli, show_wire_labels=False)
    fig2, ax2 = _render(cswap, show_wire_labels=False)
    try:
        assert _texts(ax) == []
        assert len(ax.patches) == 3
        assert _texts(ax2) == []
        assert len(ax2.patches) == 1
    finally:
        plt.close(fig)
        plt.close(fig2)


def test_swap_gate_draws_swap_markers_without_a_box() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.swap(0, 1)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert _texts(ax) == []
        assert len(ax.patches) == 0
    finally:
        plt.close(fig)


def test_unknown_multi_wire_gate_falls_back_to_a_labelled_box() -> None:
    plt = _pyplot()
    device = _LegacyDevice([{"name_or_mat": "qft", "wires": [0, 1, 2], "params": []}])

    fig, ax = _render(device, show_wire_labels=False)
    try:
        assert _texts(ax) == ["QFT"]
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)


def test_measure_allz_adds_one_marker_per_wire() -> None:
    plt = _pyplot()
    device = _LegacyDevice(
        [
            {"name_or_mat": "h", "wires": [0], "params": []},
            {"name_or_mat": "measure_allz", "wires": [], "params": []},
        ]
    )

    fig, ax = _render(device, show_wire_labels=False)
    try:
        assert _texts(ax).count("MZ") == device.n_wires
    finally:
        plt.close(fig)


def test_wire_labels_show_the_initial_state_only_when_requested() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.h(0)

    fig, ax = _render(circuit, show_initial_state=True)
    fig2, ax2 = _render(circuit)
    try:
        assert [text for text in _texts(ax) if "|0⟩" in text] == ["0: |0⟩", "1: |0⟩"]
        assert [text for text in _texts(ax2) if text in {"0", "1"}] == ["0", "1"]
    finally:
        plt.close(fig)
        plt.close(fig2)


def test_wire_labels_can_be_cropped_out_of_the_view() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.h(0)

    fig, ax = _render(circuit, show_wire_labels=False)
    try:
        assert ax.get_xlim()[0] == pytest.approx(-1.0)
        assert all("|0⟩" not in text for text in _texts(ax))
    finally:
        plt.close(fig)


def test_wire_labels_keep_the_uncropped_view_by_default() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.h(0)

    fig, ax = _render(circuit, show_wire_labels=True)
    try:
        assert ax.get_xlim()[0] == pytest.approx(-2.0)
    finally:
        plt.close(fig)


def test_wire_order_reorders_labels_and_is_copied() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(3)
    circuit.h(0)
    wire_order: list[int | str] = [2, 0, 1]

    fig, ax = _render(circuit, wire_order=wire_order, show_initial_state=True)
    try:
        assert wire_order == [2, 0, 1]
        assert MPLDrawer(circuit, wire_order=[2, 0, 1]).wire_map == {2: 0, 0: 1, 1: 2}
        assert [text for text in _texts(ax) if "|0⟩" in text] == [
            "2: |0⟩",
            "0: |0⟩",
            "1: |0⟩",
        ]
    finally:
        plt.close(fig)


def test_wire_labels_missing_from_wire_order_are_appended() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(3)
    circuit.h(0)

    drawer = MPLDrawer(circuit, wire_order=[2])
    plt.close(drawer.fig)

    assert drawer.wire_map == {2: 0, 0: 1, 1: 2}


def test_operations_without_wires_do_not_create_layers() -> None:
    plt = _pyplot()
    device = _LegacyDevice(
        [
            {"name_or_mat": "barrier", "wires": [], "params": []},
            {"name_or_mat": "h", "wires": [0], "params": []},
        ]
    )

    drawer = MPLDrawer(device)
    plt.close(drawer.fig)

    assert drawer.n_layers == 1


def test_operations_on_unknown_wires_do_not_create_layers() -> None:
    plt = _pyplot()
    device = _LegacyDevice(
        [{"name_or_mat": "h", "wires": [7], "params": []}], n_wires=2
    )

    drawer = MPLDrawer(device)
    plt.close(drawer.fig)

    assert drawer.n_layers == 0


def test_wire_count_is_detected_from_history_when_absent() -> None:
    plt = _pyplot()
    device = _LegacyDevice(
        [{"name_or_mat": "h", "wires": 2, "params": []}], n_wires=None
    )

    drawer = MPLDrawer(device)
    plt.close(drawer.fig)

    assert drawer.n_wires == 3


def test_supplied_figure_is_reused_instead_of_created() -> None:
    plt = _pyplot()
    import matplotlib.pyplot as plt_module

    circuit = fq.Circuit(1)
    circuit.h(0)
    figure = plt_module.figure(figsize=(4, 2))

    fig, ax = _render(circuit, fig=figure)
    try:
        assert fig is figure
        assert ax is figure.axes[0]
    finally:
        plt.close(fig)


def test_figsize_controls_the_created_figure() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.h(0)

    fig, _ax = _render(circuit, figsize=(9, 4))
    try:
        assert tuple(fig.get_size_inches()) == (9.0, 4.0)
    finally:
        plt.close(fig)


def test_figure_size_follows_layer_and_wire_counts_by_default() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(3)
    circuit.h(0).x(1).y(2)

    fig, _ax = _render(circuit, show_wire_labels=False)
    try:
        assert tuple(fig.get_size_inches()) == (4.0, 4.0)
    finally:
        plt.close(fig)


def test_draw_entry_points_route_the_mpl_format() -> None:
    plt = _pyplot()
    from flagquantum.drawer import draw

    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    fig, ax = draw(circuit, format="mpl", show_wire_labels=False)
    fig_from_method, ax_from_method = circuit.draw(format="mpl", show_wire_labels=False)
    try:
        assert "H" in _texts(ax)
        assert "H" in _texts(ax_from_method)
    finally:
        plt.close(fig)
        plt.close(fig_from_method)


def test_wire_options_style_the_wire_lines() -> None:
    plt = _pyplot()
    circuit = fq.Circuit(2)
    circuit.h(0)

    fig, ax = _render(
        circuit, wire_options={"color": "#123456"}, show_wire_labels=False
    )
    try:
        assert {line.get_color() for line in ax.lines} == {"#123456"}
    finally:
        plt.close(fig)
