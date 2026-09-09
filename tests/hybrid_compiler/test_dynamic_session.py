from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._hybrid import (
    BOOL,
    INDEX,
    HybridCaptureError,
    SpecializationError,
    capture_source,
    lower_dynamic_program,
    scalar_type,
    tensor_type,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.parameters import Parameter
from flagquantum.runtime.dynamic.hybrid_session import (
    execute_hybrid_dynamic_session,
)

pytestmark = pytest.mark.integration

SOURCE = """
def feedback():
    qp.H(wires=0)
    bit = qp.measure(wires=0)
    if bit:
        qp.X(wires=1)
    else:
        qp.X(wires=2)
    return bit
"""
PARAMETERIZED_SOURCE = """
def parameterized_feedback(theta, layers):
    for layer in range(layers):
        qp.RX(theta, wires=0)
    bit = qp.measure(wires=0)
    if bit:
        qp.X(wires=1)
    return bit
"""
PARAMETERIZED_PROGRAM = capture_source(
    PARAMETERIZED_SOURCE,
    (scalar_type("float64"), INDEX),
)
STATIC_BRANCH_SOURCE = """
def static_branch_feedback(use_ry, theta):
    if use_ry:
        qp.RY(theta, wires=0)
    else:
        qp.RX(theta, wires=0)
    bit = qp.measure(wires=0)
    return bit
"""
STATIC_BRANCH_PROGRAM = capture_source(
    STATIC_BRANCH_SOURCE,
    (BOOL, scalar_type("float32")),
)
LOOP_CARRIED_SOURCE = """
def loop_carried_feedback(theta, delta, layers):
    angle = theta
    wire = 0
    for layer in range(layers):
        angle = angle + delta
        wire = (wire + 1) % 2
        qp.RX(angle, wires=wire)
    qp.RY(angle, wires=2)
    bit = qp.measure(wires=1)
    return bit
"""
LOOP_CARRIED_PROGRAM = capture_source(
    LOOP_CARRIED_SOURCE,
    (scalar_type("float64"), scalar_type("float64"), INDEX),
)
BOOL_CARRIED_PROGRAM = capture_source(
    """
def bool_carried_feedback(layers):
    enabled = False
    for layer in range(layers):
        enabled = layer == 0
    if enabled:
        qp.X(wires=0)
    else:
        qp.X(wires=1)
    bit = qp.measure(wires=0)
    return bit
""",
    (INDEX,),
)


def _lowered():
    return lower_dynamic_program(capture_source(SOURCE, ()))


def test_capture_and_lower_measurement_feedback_to_existing_circuit_ir() -> None:
    lowered = _lowered()
    ir = lowered.circuit

    assert tuple(instruction.name for instruction in ir.instructions) == (
        "h",
        "measure",
        "x",
        "x",
    )
    assert ir.instructions[1].metadata == {
        "is_dynamic": True,
        "classical_bit": 0,
    }
    assert ir.instructions[2].metadata["conditions"] == ((0, 1),)
    assert ir.instructions[3].metadata["conditions"] == ((0, 0),)
    assert lowered.measurement_count == 1
    assert lowered.return_classical_bit == 0
    assert ir.metadata["hybrid_dynamic_session"] is True
    assert ir == type(ir).from_json(ir.to_json())


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_stateful_session_returns_measurements_and_continues_each_shot(
    strategy: str,
) -> None:
    result = execute_hybrid_dynamic_session(
        _lowered().circuit,
        shots=128,
        seed=17,
        strategy=strategy,
    )
    bit = result.classical_bits[:, 0]

    assert set(bit.tolist()) == {0, 1}
    assert torch.equal(result.samples[:, 0], bit)
    assert torch.equal(result.samples[:, 1], bit)
    assert torch.equal(result.samples[:, 2], 1 - bit)
    assert result.mid_circuit_measurements is result.classical_bits
    assert result.statistics["hybrid_dynamic_session"] is True
    assert result.statistics["returned_classical_bit"] == 0
    assert result.statistics["stochastic_gradient_policy"] == (
        "unsupported_fail_closed"
    )


def test_seeded_session_is_reproducible() -> None:
    first = execute_hybrid_dynamic_session(_lowered().circuit, shots=64, seed=9)
    second = execute_hybrid_dynamic_session(_lowered().circuit, shots=64, seed=9)

    assert torch.equal(first.samples, second.samples)
    assert torch.equal(first.classical_bits, second.classical_bits)


def test_dynamic_profile_fails_closed_on_unsupported_semantics() -> None:
    source = """
def conditional_measurement():
    qp.H(wires=0)
    bit = qp.measure(wires=0)
    if bit:
        nested = qp.measure(wires=1)
    return bit
"""
    with pytest.raises(SpecializationError, match="dynamic.conditional_measurement"):
        lower_dynamic_program(capture_source(source, ()))
    with pytest.raises(SpecializationError, match="expected 0 runtime input"):
        lower_dynamic_program(capture_source(SOURCE, ()), (torch.tensor(1.0),))


def test_runtime_rejects_invalid_shots_gradient_and_classical_order() -> None:
    ir = _lowered().circuit
    with pytest.raises(ValueError, match="positive integer"):
        execute_hybrid_dynamic_session(ir, shots=0)

    instructions = list(ir.instructions)
    instructions[2] = replace(
        instructions[2], params={"theta": torch.tensor(0.2, requires_grad=True)}
    )
    with pytest.raises(RuntimeError, match="stochastic gradients"):
        execute_hybrid_dynamic_session(
            replace(ir, instructions=tuple(instructions)), shots=1
        )

    invalid_order = replace(
        ir,
        instructions=(
            Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),
            *ir.instructions,
        ),
    )
    with pytest.raises(ValueError, match="before measurement"):
        execute_hybrid_dynamic_session(invalid_order, shots=1)

    invalid_condition = replace(
        ir,
        instructions=(
            *ir.instructions[:2],
            replace(ir.instructions[2], metadata={"conditions": ((0, 2),)}),
            *ir.instructions[3:],
        ),
    )
    with pytest.raises(ValueError, match="binary values"):
        execute_hybrid_dynamic_session(invalid_condition, shots=1)


@pytest.mark.parametrize(("theta", "expected"), ((0.0, 0), (torch.pi, 1)))
def test_parameterized_bounded_loop_changes_dynamic_session(
    theta: float, expected: int
) -> None:
    value = torch.tensor(theta, dtype=torch.float64)
    lowered = lower_dynamic_program(
        PARAMETERIZED_PROGRAM,
        (value, 1),
        circuit_dtype="complex128",
    )

    assert isinstance(
        lowered.circuit_template.instructions[0].params["theta"], Parameter
    )
    assert lowered.ordered_bindings[0][1] is value
    result = execute_hybrid_dynamic_session(
        lowered.bind(), shots=32, seed=5, strategy="batched"
    )

    assert torch.all(result.classical_bits[:, 0] == expected)
    assert torch.all(result.samples[:, 0] == expected)
    assert torch.all(result.samples[:, 1] == expected)


def test_dynamic_parameter_values_do_not_change_template_identity() -> None:
    first = lower_dynamic_program(
        PARAMETERIZED_PROGRAM,
        (torch.tensor(0.1, dtype=torch.float64), 2),
        circuit_dtype="complex128",
    )
    second = lower_dynamic_program(
        PARAMETERIZED_PROGRAM,
        (torch.tensor(0.9, dtype=torch.float64), 2),
        circuit_dtype="complex128",
    )

    assert first.circuit_template.content_hash == second.circuit_template.content_hash
    assert first.input_signature_identity == second.input_signature_identity
    assert first.circuit.instructions[0].params["theta"] is not (
        second.circuit.instructions[0].params["theta"]
    )


def test_parameterized_dynamic_profile_rejects_gradients_unbound_ir_and_large_loop() -> (
    None
):
    with pytest.raises(SpecializationError, match="trainable inputs"):
        lower_dynamic_program(
            PARAMETERIZED_PROGRAM,
            (torch.tensor(0.2, dtype=torch.float64, requires_grad=True), 1),
            circuit_dtype="complex128",
        )
    with pytest.raises(SpecializationError, match="unroll_limit"):
        lower_dynamic_program(
            PARAMETERIZED_PROGRAM,
            (torch.tensor(0.2, dtype=torch.float64), 5),
            circuit_dtype="complex128",
            max_unrolled_iterations=4,
        )
    lowered = lower_dynamic_program(
        PARAMETERIZED_PROGRAM,
        (torch.tensor(0.2, dtype=torch.float64), 1),
        circuit_dtype="complex128",
    )
    with pytest.raises(ValueError, match="must be bound"):
        execute_hybrid_dynamic_session(lowered.circuit_template, shots=1)


@pytest.mark.parametrize(("use_ry", "gate"), ((True, "ry"), (False, "rx")))
def test_bool_input_selects_float32_rotation_before_measurement(
    use_ry: bool, gate: str
) -> None:
    lowered = lower_dynamic_program(
        STATIC_BRANCH_PROGRAM,
        (use_ry, torch.tensor(torch.pi, dtype=torch.float32)),
        circuit_dtype="complex64",
    )

    assert lowered.circuit_template.instructions[0].name == gate
    result = execute_hybrid_dynamic_session(lowered.circuit, shots=16, seed=3)
    assert torch.all(result.classical_bits == 1)


def test_tensor_inputs_and_loop_measurement_remain_outside_profile() -> None:
    tensor_program = capture_source(
        """
def tensor_feedback(data):
    bit = qp.measure(wires=0)
    return bit
""",
        (tensor_type("float32", (1,)),),
    )
    with pytest.raises(SpecializationError, match="scalar, index, or bool"):
        lower_dynamic_program(
            tensor_program,
            (torch.zeros(1),),
            circuit_dtype="complex64",
        )

    loop_measurement = capture_source(
        """
def loop_measurement():
    for index in range(2):
        nested = qp.measure(wires=index)
    bit = qp.measure(wires=0)
    return bit
""",
        (),
    )
    with pytest.raises(SpecializationError, match="dynamic.loop_measurement"):
        lower_dynamic_program(loop_measurement)


def test_capture_and_execute_loop_carried_scalar_and_index_state() -> None:
    loop = next(
        operation
        for operation in LOOP_CARRIED_PROGRAM.body.blocks[0].operations
        if operation.name == "scf.for"
    )
    body = loop.regions[0].blocks[0]

    assert len(loop.operands) == 6  # lower, upper, step, angle, wire, effect
    assert len(loop.results) == 3
    assert len(body.arguments) == 4  # iteration, angle, wire, effect
    assert len(body.operations[-1].operands) == 3

    theta = torch.tensor(0.0, dtype=torch.float64)
    delta = torch.tensor(torch.pi, dtype=torch.float64)
    lowered = lower_dynamic_program(
        LOOP_CARRIED_PROGRAM,
        (theta, delta, 2),
        circuit_dtype="complex128",
    )

    assert tuple(item.name for item in lowered.circuit.instructions) == (
        "rx",
        "rx",
        "ry",
        "measure",
    )
    assert tuple(item.wires for item in lowered.circuit.instructions) == (
        (1,),
        (0,),
        (2,),
        (1,),
    )
    bound_angles = tuple(value for _, value in lowered.ordered_bindings)
    assert torch.equal(bound_angles[0], torch.tensor(torch.pi, dtype=torch.float64))
    assert torch.equal(bound_angles[1], torch.tensor(2 * torch.pi, dtype=torch.float64))
    assert bound_angles[2] is bound_angles[1]

    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=32, seed=13, strategy="batched"
    )
    assert torch.all(result.classical_bits[:, 0] == 1)
    assert torch.all(result.samples[:, 1] == 1)


@pytest.mark.parametrize(("layers", "expected"), ((0, 0), (1, 1), (2, 0)))
def test_loop_carried_bool_selects_post_loop_branch(layers: int, expected: int) -> None:
    lowered = lower_dynamic_program(BOOL_CARRIED_PROGRAM, (layers,))
    result = execute_hybrid_dynamic_session(lowered.circuit, shots=8, seed=2)

    assert torch.all(result.classical_bits[:, 0] == expected)
    assert torch.all(result.samples[:, 0] == expected)


def test_loop_carry_rejects_tensor_state_and_target_shadowing() -> None:
    tensor_carry = """
def tensor_carry(data):
    for layer in range(1):
        data = data
    bit = qp.measure(wires=0)
    return bit
"""
    with pytest.raises(HybridCaptureError, match="classical_carry_type"):
        capture_source(tensor_carry, (tensor_type("float32", (1,)),))

    target_shadow = """
def target_shadow(layer):
    for layer in range(1):
        qp.X(wires=0)
    bit = qp.measure(wires=0)
    return bit
"""
    with pytest.raises(HybridCaptureError, match="control.target_shadow"):
        capture_source(target_shadow, (INDEX,))

    nested_carry = """
def nested_carry(theta):
    angle = theta
    for outer in range(1):
        for inner in range(1):
            angle = angle + theta
        qp.RX(angle, wires=0)
    bit = qp.measure(wires=0)
    return bit
"""
    with pytest.raises(HybridCaptureError, match="control.classical_carry"):
        capture_source(nested_carry, (scalar_type("float64"),))
