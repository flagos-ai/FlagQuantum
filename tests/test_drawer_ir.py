from typing import ClassVar

import pytest

import flagquantum as fq
from flagquantum.drawer import draw
from flagquantum.drawer.ir_adapter import to_drawable_circuit

pytestmark = pytest.mark.integration


def test_draw_accepts_native_circuit_and_ir():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    circuit_text = draw(circuit)
    ir_text = draw(circuit.to_ir())

    assert circuit_text == ir_text
    assert "H" in circuit_text
    assert "X" in circuit_text


def test_circuit_draw_uses_unified_drawer():
    circuit = fq.Circuit(2)
    circuit.rx(0, theta=0.25).cz(0, 1)

    text = circuit.draw(decimals=2)

    assert text == draw(circuit.to_ir(), decimals=2)
    assert "RX(0.25)" in text
    assert "Z" in text


def test_drawer_keeps_legacy_qdev_compatibility():
    class LegacyDevice:
        n_wires = 1
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "wires": [0], "params": []}
        ]

    assert "H" in draw(LegacyDevice())


@pytest.mark.parametrize("format", ["text", "mpl"])
def test_drawer_accepts_a_legacy_qdev_on_both_formats(format: str) -> None:
    """A qdev that predates the qubit vocabulary draws on either renderer.

    ``n_wires`` and ``wires`` are the only spellings a legacy object knows, so
    the adapter has to translate both. The width is read from ``n_wires`` and
    not inferred, because inference alone would collapse a three-qubit device
    that happens to touch qubit 0 first.
    """

    class LegacyDevice:
        n_wires = 3
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "wires": [0], "params": []},
            {"name_or_mat": "cx", "wires": [0, 2], "params": []},
        ]

    if format == "mpl":
        pytest.importorskip("matplotlib")
        figure, _ = draw(LegacyDevice(), format="mpl", show_all_qubits=True)
        assert figure.axes
    else:
        text = draw(LegacyDevice(), show_all_qubits=True)
        assert "H" in text
        assert "X" in text


def test_drawer_infers_the_width_when_a_legacy_qdev_omits_it() -> None:
    """A legacy qdev that reports no width at all still gets its qubits drawn.

    This is the shape the adapter's inference exists for: the object names its
    qubits only through its operation history, and an entry may name a single
    qubit without wrapping it in a list.
    """

    class WidthlessDevice:
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "x", "wires": 2, "params": []}
        ]

    text = draw(WidthlessDevice(), show_all_qubits=True)

    assert "X" in text
    # Three qubits were inferred, so the diagram has a row for each of them.
    assert len([line for line in text.splitlines() if line.strip()]) >= 3


def test_drawer_normalizes_a_legacy_entry_before_a_renderer_sees_it() -> None:
    """The adapter is the only door; renderers never meet a legacy spelling.

    This is the property whose absence caused the regression: both renderers
    read ``op["qubits"]``, so an entry that reached them still spelled
    ``wires`` was read as occupying no qubits at all and the text drawer then
    called ``min()`` on an empty sequence. Asserting the normalization directly
    pins the contract instead of only its symptom.
    """

    class LegacyDevice:
        n_wires = 2
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "wires": [1], "params": []}
        ]

    adapted = to_drawable_circuit(LegacyDevice())

    assert adapted.n_qubits == 2
    assert adapted.op_history[0]["qubits"] == [1]
    assert "wires" not in adapted.op_history[0]


def test_drawer_normalization_does_not_mutate_the_caller_entry() -> None:
    """Normalizing copies: the caller's own operation history is left alone."""

    class LegacyDevice:
        n_wires = 1
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "wires": [0], "params": []}
        ]

    adapted = to_drawable_circuit(LegacyDevice())

    assert LegacyDevice.op_history[0] == {
        "name_or_mat": "h",
        "wires": [0],
        "params": [],
    }
    assert adapted.op_history[0] is not LegacyDevice.op_history[0]


def test_drawer_accepts_a_legacy_entry_on_a_qubit_named_device() -> None:
    """The two spellings are independent, so a half-migrated object works too.

    An object can report ``n_qubits`` and still carry ``wires``-keyed entries.
    Reading one spelling per object rather than one per concern would miss it.
    """

    class HalfMigrated:
        n_qubits = 2
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "wires": [1], "params": []}
        ]

    assert "H" in draw(HalfMigrated(), show_all_qubits=True)


def test_drawer_prefers_the_qubit_spelling_when_an_entry_carries_both() -> None:
    """A qubit-named entry wins, so the adapter never overrides our own data."""

    class Both:
        n_qubits = 3
        op_history: ClassVar[list[dict[str, object]]] = [
            {"name_or_mat": "h", "qubits": [2], "wires": [0], "params": []}
        ]

    adapted = to_drawable_circuit(Both())

    assert adapted.op_history[0]["qubits"] == [2]


def test_native_circuit_still_draws_after_the_adapter_change() -> None:
    """The native path is unchanged: a Circuit and its IR render identically."""

    circuit = fq.Circuit(3).h(0).cx(0, 2)

    assert draw(circuit, show_all_qubits=True) == draw(
        circuit.to_ir(), show_all_qubits=True
    )
