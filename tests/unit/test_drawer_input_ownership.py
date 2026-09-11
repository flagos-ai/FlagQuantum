"""Drawing must leave caller-owned presentation options reusable."""

from copy import deepcopy

import pytest

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.drawer.text_drawer import TextDrawer

pytestmark = pytest.mark.unit


def test_text_drawer_copies_wire_order() -> None:
    program = CircuitIR(n_wires=2, instructions=(Instruction("h", (0,)),))
    wire_order = [1, 0]
    drawer = TextDrawer(program, wire_order=wire_order, show_all_wires=True)

    wire_order.reverse()

    assert drawer.wire_order == [1, 0]
    assert "H" in drawer.draw()


def test_mpl_options_can_be_reused_without_mutation() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from flagquantum.drawer.mpl_drawer import draw_mpl

    program = CircuitIR(n_wires=2, instructions=(Instruction("h", (0,)),))
    wire_order = [0]
    labels = {"color": "navy", "show_initial_state": False}
    original_labels = deepcopy(labels)

    fig, ax = draw_mpl(
        program,
        wire_order=wire_order,
        label_options=labels,
        show_initial_state=True,
    )
    try:
        assert wire_order == [0]
        assert labels == original_labels
        assert any(text.get_text() == "0: |0⟩" for text in ax.texts)
        fig.canvas.draw()
    finally:
        plt.close(fig)
