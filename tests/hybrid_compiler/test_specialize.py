from __future__ import annotations

import pytest
import torch

from flagquantum.compiler._hybrid import (
    BOOL,
    INDEX,
    SpecializationError,
    capture_source,
    specialize_program,
    tensor_type,
)

pytestmark = pytest.mark.unit

WEIGHTS = tensor_type("float64", (None, 4))
DATA = tensor_type("float64", (4,))

SOURCE = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    for row in weights:
        for wire, parameter in enumerate(row):
            if parameter > 0:
                qp.RX(parameter, wires=wire)
            elif parameter < 0:
                qp.RY(parameter, wires=wire)
        for wire in range(4):
            qp.CNOT(wires=[wire, jnp.mod(wire + 1, 4)])
    return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))
"""


def program():
    return capture_source(SOURCE, (WEIGHTS, DATA))


@pytest.mark.parametrize(
    ("weights", "selected"),
    (
        ([0.2, 0.3, 0.4, 0.5], ("rx", "rx", "rx", "rx")),
        ([-0.2, -0.3, -0.4, -0.5], ("ry", "ry", "ry", "ry")),
        ([0.0, 0.0, 0.0, 0.0], ()),
        ([0.2, -0.3, 0.0, 0.5], ("rx", "ry", "rx")),
    ),
)
def test_golden_branch_patterns_select_expected_gate_trace(
    weights: list[float], selected: tuple[str, ...]
) -> None:
    runtime_weights = torch.tensor([weights], dtype=torch.float64)
    data = torch.arange(4, dtype=torch.float64)

    trace = specialize_program(program(), (runtime_weights, data))

    assert tuple(gate.name for gate in trace.gates[:4]) == ("rx",) * 4
    assert tuple(gate.name for gate in trace.gates[4 : 4 + len(selected)]) == selected
    assert tuple(gate.name for gate in trace.gates[-4:]) == ("cx",) * 4
    assert tuple(gate.wires for gate in trace.gates[-4:]) == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    )
    assert trace.observables == (("z", 0), ("z", 3))


def test_parameter_values_do_not_change_selected_structure_identity() -> None:
    data_a = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    data_b = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    weights_a = torch.tensor([[0.2, -0.3, 0.0, 0.5]], dtype=torch.float64)
    weights_b = torch.tensor([[2.0, -3.0, 0.0, 5.0]], dtype=torch.float64)

    first = specialize_program(program(), (weights_a, data_a))
    second = specialize_program(program(), (weights_b, data_b))

    assert first.input_signature_identity == second.input_signature_identity
    assert first.structure_identity == second.structure_identity


def test_branch_change_changes_selected_structure_identity() -> None:
    data = torch.zeros(4, dtype=torch.float64)
    positive = torch.ones((1, 4), dtype=torch.float64)
    negative = -positive

    first = specialize_program(program(), (positive, data))
    second = specialize_program(program(), (negative, data))

    assert first.structure_identity != second.structure_identity


def test_ordered_comparison_boundary_is_observable_and_optionally_rejected() -> None:
    data = torch.zeros(4, dtype=torch.float64)
    weights = torch.zeros((1, 4), dtype=torch.float64)

    trace = specialize_program(program(), (weights, data))

    assert trace.nonsmooth_control_decisions
    assert set(trace.nonsmooth_control_decisions) == {
        "gt:equality_boundary",
        "lt:equality_boundary",
    }
    with pytest.raises(SpecializationError, match="gradient.nonsmooth_control"):
        specialize_program(
            program(),
            (weights, data),
            require_smooth_gradients=True,
        )


def test_runtime_input_signature_is_validated_before_specialization() -> None:
    invalid_data = torch.zeros(5, dtype=torch.float64)
    weights = torch.zeros((1, 4), dtype=torch.float64)

    with pytest.raises(SpecializationError, match="input.shape"):
        specialize_program(program(), (weights, invalid_data))


def test_total_loop_unrolling_is_bounded() -> None:
    data = torch.zeros(4, dtype=torch.float64)
    weights = torch.ones((2, 4), dtype=torch.float64)

    with pytest.raises(SpecializationError, match="control.unroll_limit"):
        specialize_program(program(), (weights, data), max_unrolled_iterations=4)


def test_dynamic_zero_loop_step_fails_before_lowering() -> None:
    source = """
def invalid(step, data):
    qp.AngleEmbedding(data, wires=range(4))
    for wire in range(0, 4, step):
        qp.RX(0.1, wires=wire)
    return qp.expval(qp.PauliZ(0))
"""
    captured = capture_source(
        source,
        (
            INDEX,
            DATA,
        ),
    )

    with pytest.raises(SpecializationError, match="control.range_step"):
        specialize_program(
            captured,
            (
                torch.tensor(0, dtype=torch.int64),
                torch.zeros(4, dtype=torch.float64),
            ),
        )


def test_index_input_rejects_floating_tensor_without_truncation() -> None:
    source = """
def indexed(wire, data):
    qp.AngleEmbedding(data, wires=range(4))
    qp.RX(0.1, wires=wire)
    return qp.expval(qp.PauliZ(0))
"""
    captured = capture_source(source, (INDEX, DATA))

    with pytest.raises(SpecializationError, match="input.index"):
        specialize_program(
            captured,
            (
                torch.tensor(1.5),
                torch.zeros(4, dtype=torch.float64),
            ),
        )


@pytest.mark.parametrize(
    ("first", "second", "gate"),
    ((True, False, "x"), (True, True, "h"), (False, False, "h")),
)
def test_static_boolean_not_and_conjunction_specialize(
    first: bool, second: bool, gate: str
) -> None:
    captured = capture_source(
        """
def boolean_select(first, second):
    if first and not second:
        qp.X(wires=0)
    else:
        qp.H(wires=0)
    return qp.expval(qp.PauliZ(0))
""",
        (BOOL, BOOL),
    )

    trace = specialize_program(captured, (first, second))

    assert tuple(item.name for item in trace.gates) == (gate,)
