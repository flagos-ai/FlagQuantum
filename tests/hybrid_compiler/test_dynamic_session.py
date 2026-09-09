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
BRANCH_CARRIED_PROGRAM = capture_source(
    """
def branch_carried_feedback(select_first, theta):
    angle = theta
    wire = 0
    enabled = False
    if select_first:
        angle = theta + theta
        wire = 1
        enabled = True
    else:
        angle = theta
        wire = 0
        enabled = False
    if enabled:
        qp.RX(angle, wires=wire)
    else:
        qp.RY(angle, wires=wire)
    bit = qp.measure(wires=wire)
    return bit
""",
    (BOOL, scalar_type("float64")),
)
LOOP_BRANCH_CARRIED_PROGRAM = capture_source(
    """
def loop_branch_carried_feedback(theta, delta, layers):
    angle = theta
    wire = 0
    enabled = False
    for layer in range(layers):
        if layer % 2 == 0:
            angle = angle + delta
            wire = (wire + 1) % 2
            enabled = True
        else:
            angle = angle + delta
            enabled = False
        if enabled:
            qp.RX(angle, wires=wire)
        else:
            qp.RY(angle, wires=wire)
    bit = qp.measure(wires=wire)
    return bit
""",
    (scalar_type("float64"), scalar_type("float64"), INDEX),
)
BRANCH_LOOP_CARRIED_PROGRAM = capture_source(
    """
def branch_loop_carried_feedback(enabled, theta, delta):
    angle = theta
    if enabled:
        for layer in range(2):
            angle = angle + delta
    qp.RX(angle, wires=0)
    bit = qp.measure(wires=0)
    return bit
""",
    (BOOL, scalar_type("float64"), scalar_type("float64")),
)
MEASUREMENT_CONJUNCTION_PROGRAM = capture_source(
    """
def measurement_conjunction():
    qp.H(wires=0)
    qp.H(wires=1)
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    if first and not second:
        qp.X(wires=2)
    return first
""",
    (),
)
MEASUREMENT_COMPARISON_PROGRAM = capture_source(
    """
def measurement_comparison():
    qp.H(wires=0)
    bit = qp.measure(wires=0)
    if bit == False:
        qp.X(wires=1)
    else:
        qp.X(wires=2)
    return bit
""",
    (),
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

    malformed_clauses = replace(
        ir,
        instructions=(
            *ir.instructions[:2],
            replace(
                ir.instructions[2],
                metadata={"condition_clauses": (((0, 1), (0, 0)), ((0, 1),))},
            ),
            *ir.instructions[3:],
        ),
    )
    with pytest.raises(ValueError, match="unique bits"):
        execute_hybrid_dynamic_session(malformed_clauses, shots=1)

    nonminimal_clauses = replace(
        ir,
        instructions=(
            *ir.instructions[:2],
            replace(
                ir.instructions[2],
                metadata={"condition_clauses": (((0, 1),), ((0, 1), (1, 0)))},
            ),
            *ir.instructions[3:],
        ),
    )
    with pytest.raises(ValueError, match="canonical, distinct, and minimal"):
        execute_hybrid_dynamic_session(nonminimal_clauses, shots=1)


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


def test_tensor_inputs_remain_outside_dynamic_profile() -> None:
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


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_fixed_round_syndrome_measurement_feedback_and_reset(strategy: str) -> None:
    syndrome_program = capture_source(
        """
def fixed_round_correction():
    qp.X(wires=0)
    syndrome = False
    for round_id in range(3):
        qp.CNOT(wires=[0, 1])
        syndrome = qp.measure(wires=1)
        if syndrome:
            qp.X(wires=0)
        qp.reset(wires=1)
    return syndrome
""",
        (),
    )
    lowered = lower_dynamic_program(syndrome_program)
    measurements = tuple(
        instruction
        for instruction in lowered.circuit.instructions
        if instruction.name == "measure"
    )
    resets = tuple(
        instruction
        for instruction in lowered.circuit.instructions
        if instruction.name == "reset"
    )

    assert lowered.measurement_count == 3
    assert lowered.return_classical_bit == 2
    assert tuple(item.metadata["classical_bit"] for item in measurements) == (0, 1, 2)
    assert len(resets) == 3
    assert all(item.metadata == {"is_dynamic": True} for item in resets)

    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=32, seed=47, strategy=strategy
    )
    assert torch.all(result.classical_bits[:, 0] == 1)
    assert torch.all(result.classical_bits[:, 1:] == 0)
    assert torch.all(result.samples[:, 0] == 0)
    assert torch.all(result.samples[:, 1] == 0)

    with pytest.raises(SpecializationError, match="exceeds 2 measurements"):
        lower_dynamic_program(syndrome_program, max_dynamic_measurements=2)


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


@pytest.mark.parametrize(
    ("select_first", "gate", "wire", "expected"),
    ((True, "rx", 1, 0), (False, "ry", 0, 1)),
)
def test_capture_and_execute_branch_carried_classical_state(
    select_first: bool, gate: str, wire: int, expected: int
) -> None:
    branch = next(
        operation
        for operation in BRANCH_CARRIED_PROGRAM.body.blocks[0].operations
        if operation.name == "scf.if"
    )

    assert len(branch.operands) == 5  # predicate, angle, wire, bool, effect
    assert len(branch.results) == 4
    for region in branch.regions:
        block = region.blocks[0]
        assert len(block.arguments) == 4
        assert len(block.operations[-1].operands) == 4

    lowered = lower_dynamic_program(
        BRANCH_CARRIED_PROGRAM,
        (select_first, torch.tensor(torch.pi, dtype=torch.float64)),
        circuit_dtype="complex128",
    )
    rotation = lowered.circuit.instructions[0]

    assert rotation.name == gate
    assert rotation.wires == (wire,)
    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=16, seed=7, strategy="batched"
    )
    assert torch.all(result.classical_bits[:, 0] == expected)
    assert torch.all(result.samples[:, wire] == expected)


def test_branch_carry_rejects_tensor_state() -> None:
    tensor_carry = """
def tensor_branch(flag, data):
    if flag:
        data = data
    bit = qp.measure(wires=0)
    return bit
"""
    with pytest.raises(HybridCaptureError, match="classical_carry_type"):
        capture_source(tensor_carry, (BOOL, tensor_type("float32", (1,))))


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_measurement_branch_carries_scalar_index_and_bool_state(strategy: str) -> None:
    measurement_carry = capture_source(
        """
def measurement_branch(zero, half_turn, delta):
    qp.H(wires=0)
    angle = zero
    wire = 1
    enabled = False
    bit = qp.measure(wires=0)
    if bit:
        angle = half_turn
        wire = 2
        enabled = True
    for layer in range(2):
        angle = angle + delta
    if enabled:
        qp.X(wires=3)
    if angle > zero:
        qp.X(wires=4)
    qp.RX(angle, wires=wire)
    return bit
""",
        (
            scalar_type("float64"),
            scalar_type("float64"),
            scalar_type("float64"),
        ),
    )
    lowered = lower_dynamic_program(
        measurement_carry,
        (
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(torch.pi, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        ),
        circuit_dtype="complex128",
    )
    rotations = tuple(
        instruction
        for instruction in lowered.circuit.instructions
        if instruction.name == "rx"
    )
    assert tuple((item.wires, item.metadata["conditions"]) for item in rotations) == (
        ((2,), ((0, 1),)),
        ((1,), ((0, 0),)),
    )
    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=128, seed=43, strategy=strategy
    )
    bit = result.classical_bits[:, 0]
    assert set(bit.tolist()) == {0, 1}
    assert torch.equal(result.samples[:, 1], torch.zeros_like(bit))
    assert torch.equal(result.samples[:, 2], bit)
    assert torch.equal(result.samples[:, 3], bit)
    assert torch.equal(result.samples[:, 4], bit)


def test_measurement_dependent_value_cases_obey_configured_ceiling() -> None:
    program = capture_source(
        """
def value_case_limit():
    wire = 0
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    third = qp.measure(wires=2)
    if first:
        wire = 1
    if second:
        wire = 2
    if third:
        wire = 3
    qp.X(wires=wire)
    return first
""",
        (),
    )

    with pytest.raises(SpecializationError, match="value expansion exceeds 3 cases"):
        lower_dynamic_program(program, max_condition_clauses=3)


def test_equal_measurement_dependent_cases_coalesce_to_unconditional_value() -> None:
    program = capture_source(
        """
def equal_cases():
    wire = 1
    bit = qp.measure(wires=0)
    if bit:
        wire = 1
    qp.X(wires=wire)
    return bit
""",
        (),
    )

    gate = lower_dynamic_program(program).circuit.instructions[-1]
    assert gate.wires == (1,)
    assert gate.metadata == {}


def test_measurement_dependent_loop_bound_fails_closed() -> None:
    program = capture_source(
        """
def dynamic_bound():
    layers = 1
    bit = qp.measure(wires=0)
    if bit:
        layers = 2
    for layer in range(layers):
        qp.X(wires=1)
    return bit
""",
        (),
    )

    with pytest.raises(SpecializationError, match="measurement-dependent loop bounds"):
        lower_dynamic_program(program)


def test_loop_carries_state_through_nested_branches() -> None:
    outer_loop = next(
        operation
        for operation in LOOP_BRANCH_CARRIED_PROGRAM.body.blocks[0].operations
        if operation.name == "scf.for"
    )
    outer_body = outer_loop.regions[0].blocks[0]
    inner_branch = next(
        operation for operation in outer_body.operations if operation.name == "scf.if"
    )

    assert len(outer_loop.operands) == 7
    assert len(outer_loop.results) == 4
    assert len(outer_body.arguments) == 5
    assert len(inner_branch.operands) == 5
    assert len(inner_branch.results) == 4

    lowered = lower_dynamic_program(
        LOOP_BRANCH_CARRIED_PROGRAM,
        (
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(torch.pi, dtype=torch.float64),
            2,
        ),
        circuit_dtype="complex128",
    )

    assert tuple(item.name for item in lowered.circuit.instructions) == (
        "rx",
        "ry",
        "measure",
    )
    assert tuple(item.wires for item in lowered.circuit.instructions) == (
        (1,),
        (1,),
        (1,),
    )
    result = execute_hybrid_dynamic_session(lowered.circuit, shots=16, seed=11)
    assert torch.all(result.classical_bits[:, 0] == 1)


@pytest.mark.parametrize(("enabled", "expected"), ((True, 1), (False, 0)))
def test_branch_carries_state_through_nested_loop(enabled: bool, expected: int) -> None:
    lowered = lower_dynamic_program(
        BRANCH_LOOP_CARRIED_PROGRAM,
        (
            enabled,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(torch.pi / 2, dtype=torch.float64),
        ),
        circuit_dtype="complex128",
    )
    result = execute_hybrid_dynamic_session(lowered.circuit, shots=16, seed=19)

    assert torch.all(result.classical_bits[:, 0] == expected)


def test_nested_loop_still_obeys_cumulative_unroll_ceiling() -> None:
    with pytest.raises(SpecializationError, match="control.unroll_limit"):
        lower_dynamic_program(
            BRANCH_LOOP_CARRIED_PROGRAM,
            (
                True,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(torch.pi / 2, dtype=torch.float64),
            ),
            circuit_dtype="complex128",
            max_unrolled_iterations=1,
        )


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_measurement_negation_and_conjunction_control_gate(
    strategy: str,
) -> None:
    lowered = lower_dynamic_program(MEASUREMENT_CONJUNCTION_PROGRAM)
    controlled = lowered.circuit.instructions[-1]

    assert controlled.name == "x"
    assert controlled.metadata["conditions"] == ((0, 1), (1, 0))
    result = execute_hybrid_dynamic_session(
        lowered.circuit,
        shots=128,
        seed=23,
        strategy=strategy,
    )
    first = result.classical_bits[:, 0]
    second = result.classical_bits[:, 1]
    expected = first * (1 - second)

    assert set(zip(first.tolist(), second.tolist())) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert torch.equal(result.samples[:, 2], expected)


def test_measurement_bool_comparison_controls_both_branches() -> None:
    lowered = lower_dynamic_program(MEASUREMENT_COMPARISON_PROGRAM)
    first_gate, second_gate = lowered.circuit.instructions[-2:]

    assert first_gate.metadata["conditions"] == ((0, 0),)
    assert second_gate.metadata["conditions"] == ((0, 1),)
    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=128, seed=29, strategy="batched"
    )
    bit = result.classical_bits[:, 0]

    assert set(bit.tolist()) == {0, 1}
    assert torch.equal(result.samples[:, 1], 1 - bit)
    assert torch.equal(result.samples[:, 2], bit)

    inequality = capture_source(
        """
def measurement_inequality():
    bit = qp.measure(wires=0)
    if bit != True:
        qp.X(wires=1)
    else:
        qp.X(wires=2)
    return bit
""",
        (),
    )
    inequality_ir = lower_dynamic_program(inequality).circuit
    assert inequality_ir.instructions[-2].metadata["conditions"] == ((0, 0),)
    assert inequality_ir.instructions[-1].metadata["conditions"] == ((0, 1),)


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_measurement_disjunction_controls_gate(strategy: str) -> None:
    program = capture_source(
        """
def disjunction():
    qp.H(wires=0)
    qp.H(wires=1)
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    if first or second:
        qp.X(wires=2)
    return first
""",
        (),
    )
    lowered = lower_dynamic_program(program)
    assert lowered.circuit.instructions[-1].metadata["condition_clauses"] == (
        ((0, 1),),
        ((1, 1),),
    )
    result = execute_hybrid_dynamic_session(
        lowered.circuit, shots=128, seed=31, strategy=strategy
    )
    first = result.classical_bits[:, 0]
    second = result.classical_bits[:, 1]
    assert torch.equal(result.samples[:, 2], first | second)


def test_negated_conjunction_and_quantum_else_are_lowered_exactly() -> None:
    program = capture_source(
        """
def negated_conjunction():
    qp.H(wires=0)
    qp.H(wires=1)
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    if not (first and second):
        qp.X(wires=2)
    else:
        qp.X(wires=3)
    return first
""",
        (),
    )
    ir = lower_dynamic_program(program).circuit
    assert ir.instructions[-2].metadata["condition_clauses"] == (
        ((0, 0),),
        ((1, 0),),
    )
    assert ir.instructions[-1].metadata["conditions"] == ((0, 1), (1, 1))
    result = execute_hybrid_dynamic_session(ir, shots=128, seed=37, strategy="batched")
    expected = result.classical_bits[:, 0] & result.classical_bits[:, 1]
    assert torch.equal(result.samples[:, 2], 1 - expected)
    assert torch.equal(result.samples[:, 3], expected)


@pytest.mark.parametrize(
    ("operator", "clauses", "equal"),
    (
        ("==", (((0, 0), (1, 0)), ((0, 1), (1, 1))), True),
        ("!=", (((0, 0), (1, 1)), ((0, 1), (1, 0))), False),
    ),
)
def test_measurement_to_measurement_comparison(operator, clauses, equal) -> None:
    program = capture_source(
        f"""
def compare_measurements():
    qp.H(wires=0)
    qp.H(wires=1)
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    if first {operator} second:
        qp.X(wires=2)
    return first
""",
        (),
    )
    ir = lower_dynamic_program(program).circuit
    assert ir.instructions[-1].metadata["condition_clauses"] == clauses
    result = execute_hybrid_dynamic_session(ir, shots=128, seed=41, strategy="batched")
    expected = result.classical_bits[:, 0] == result.classical_bits[:, 1]
    if not equal:
        expected = ~expected
    assert torch.equal(result.samples[:, 2].bool(), expected)


def test_measurement_predicates_are_minimized_and_bounded() -> None:
    minimized = capture_source(
        """
def minimized():
    first = qp.measure(wires=0)
    second = qp.measure(wires=1)
    if first or (first and second):
        qp.X(wires=2)
    return first
""",
        (),
    )
    gate = lower_dynamic_program(minimized).circuit.instructions[-1]
    assert gate.metadata["conditions"] == ((0, 1),)

    explosive = capture_source(
        """
def explosive():
    a = qp.measure(wires=0)
    b = qp.measure(wires=1)
    c = qp.measure(wires=2)
    d = qp.measure(wires=3)
    if (a or b) and (c or d):
        qp.X(wires=4)
    return a
""",
        (),
    )
    with pytest.raises(SpecializationError, match="exceeds 3 canonical clauses"):
        lower_dynamic_program(explosive, max_condition_clauses=3)


def test_measurement_boolean_profile_still_rejects_inline_measurement() -> None:

    inline_measurement = """
def inline_measurement():
    first = qp.measure(wires=0)
    if first and qp.measure(wires=1):
        qp.X(wires=2)
    return first
"""
    with pytest.raises(HybridCaptureError, match="must be assigned"):
        capture_source(inline_measurement, ())
