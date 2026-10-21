"""Declared layer continuity, trainability and circuit-panel semantics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.drawer import ClassicalLayer, draw_hybrid

pytestmark = pytest.mark.unit


def cnn_layers() -> tuple[ClassicalLayer, ...]:
    return (
        ClassicalLayer("Conv2d", (1, 8, 8), (3, 8, 8), True),
        ClassicalLayer("Pool", (3, 8, 8), (3, 1, 1)),
        ClassicalLayer("Flatten", (3, 1, 1), (3,)),
    )


def test_declared_cnn_shape_checks_and_frozen_stage_ownership() -> None:
    layers = list(cnn_layers())
    head = [ClassicalLayer("Linear", (3,), (2,), True)]
    circuit = fq.Circuit(3).ry(0, fq.Parameter("theta_0"))
    original = circuit.to_ir().to_dict()
    figure = draw_hybrid(
        circuit,
        encoder=layers,
        head=head,
        readout="Z expectations",
        readout_features=3,
        training=True,
        encoder_trainable=False,
    )
    assert "Conv2d" in figure.source and "Flatten" in figure.source
    assert "encoder update" not in figure.source
    assert "quantum update" in figure.source and "head update" in figure.source
    assert "Z expectations" in figure.source and "3 values" in figure.source
    before = figure.source
    layers.clear()
    head.clear()
    assert figure.source == before
    assert circuit.to_ir().to_dict() == original
    with pytest.raises(FrozenInstanceError):
        cnn_layers()[0].trainable = False


@pytest.mark.parametrize("shape", [(), (0,), (-1, 2), (True,), (1.5,)])
def test_invalid_shapes_fail_before_rendering(shape: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ClassicalLayer("Conv", shape, (3,))


def test_shape_continuity_and_quantum_boundary_conflicts() -> None:
    circuit = fq.Circuit(3)
    with pytest.raises(ValueError, match="shape mismatch"):
        draw_hybrid(
            circuit, encoder=[cnn_layers()[0], ClassicalLayer("Flatten", (2,), (3,))]
        )
    with pytest.raises(ValueError, match="conflicts"):
        draw_hybrid(circuit, encoder=cnn_layers(), encoder_features=4)
    with pytest.raises(ValueError, match="final vector"):
        draw_hybrid(circuit, encoder=[cnn_layers()[0]])
    head = [ClassicalLayer("Linear", (3,), (2,))]
    with pytest.raises(ValueError, match="requires readout"):
        draw_hybrid(circuit, head=head)
    with pytest.raises(ValueError, match="head input"):
        draw_hybrid(circuit, head=head, readout="Z", readout_features=2)
    with pytest.raises(ValueError, match="head_outputs"):
        draw_hybrid(circuit, head=head, readout="Z", readout_features=3, head_outputs=3)


def test_unknown_counts_and_readout_are_not_filled_with_example_values() -> None:
    source = draw_hybrid(fq.Circuit(2).h(0), training=True).source
    assert "unspecified features" in source and "unspecified outputs" in source
    assert "Readout unspecified" in source
    assert r"\fhConnect[fh/measurement]" not in source
    assert "quantum update" not in source  # No tunable gates exist.
    explicit = draw_hybrid(
        fq.Circuit(2), encoder_features=32, head_outputs=10, readout="Z_0%"
    ).source
    assert "32 features" in explicit and "10 outputs" in explicit
    assert r"Z\_0\%" in explicit


def test_nonparametric_layers_and_frozen_quantum_have_no_feedback() -> None:
    source = draw_hybrid(
        fq.Circuit(3).ry(0, 0.2),
        encoder=cnn_layers()[1:],
        head=[ClassicalLayer("Linear", (3,), (2,))],
        readout="Z",
        readout_features=3,
        training=True,
        quantum_trainable=False,
    ).source
    assert "No trainable stages" in source
    assert " update}" not in source


@pytest.mark.parametrize("view", ["auto", "overview", "detail"])
def test_circuit_panels_preserve_order_and_terminal_requests_once(view: str) -> None:
    ir = CircuitIR(
        2,
        tuple(Instruction("ry", (i % 2,), {"theta": float(i)}) for i in range(13)),
        observables=(ObservableNode("z", (0,)),),
        measurements=(MeasurementNode("expectation_z", (0, 1)),),
    )
    source = draw_hybrid(ir, quantum_view=view, gates_per_panel=5).source
    assert r"\resizebox" not in source
    if view == "overview":
        assert "13 gates" in source and r"\textbf{b}\quad Quantum circuit" not in source
    else:
        positions = [source.index(rf"R_y({i})") for i in range(13)]
        assert positions == sorted(positions)
        assert source.count("Observable:") == source.count("Readout:") == 1
        assert "gates 1--5" in source and "gates 11--13" in source
        assert source.count(r"qubit {$q_{\idx}$} q[2];") == 3
    assert ir.instructions[0].params["theta"] == 0.0


def test_short_empty_and_unsupported_circuits_in_all_modes() -> None:
    short = fq.Circuit(2).h(0).cx(0, 1)
    assert "cnot q[1]" in draw_hybrid(short).source
    assert r"\textbf{b}\quad Quantum circuit" not in draw_hybrid(short).source
    assert "no gates" in draw_hybrid(fq.Circuit(2), quantum_view="detail").source
    ir = replace(
        short.to_ir(),
        instructions=(Instruction("bit_flip", (0,), {"probability": 0.1}),),
    )
    with pytest.raises(ValueError, match="does not support"):
        draw_hybrid(ir, quantum_view="overview")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": ""},
        {"encoder": []},
        {"head": [object()]},
        {"readout": ""},
        {"readout_features": 3},
        {"readout": "Z", "readout_features": 0},
        {"quantum_view": "tiny"},
        {"gates_per_panel": 0},
        {"gates_per_panel": True},
        {"encoder_trainable": 1},
        {"quantum_trainable": 1},
        {"head_trainable": 1},
    ],
)
def test_invalid_layout_inputs_fail_before_a_compiler_is_needed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        draw_hybrid(fq.Circuit(2), **kwargs)


def test_layer_labels_and_trainability_are_validated_and_escaped() -> None:
    for label in ("", " "):
        with pytest.raises(ValueError):
            ClassicalLayer(label, (1,), (1,))
    with pytest.raises(TypeError):
        ClassicalLayer("Dense", (1,), (1,), trainable=1)
    layer = ClassicalLayer(r"Conv_1%\input{secret}", (1,), (2,))
    source = draw_hybrid(fq.Circuit(2), encoder=[layer]).source
    assert r"\input{secret}" not in source
    assert r"Conv\_1\%\textbackslash{}input\{secret\}" in source
