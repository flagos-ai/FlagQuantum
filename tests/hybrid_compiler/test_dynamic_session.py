from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._hybrid import (
    SpecializationError,
    capture_source,
    lower_dynamic_program,
)
from flagquantum.core.ir import Instruction
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
    with pytest.raises(SpecializationError, match="accepts no runtime inputs"):
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
