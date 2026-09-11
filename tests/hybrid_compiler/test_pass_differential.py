from __future__ import annotations

import random

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._hybrid import (
    INDEX,
    SpecializationError,
    capture_source,
    run_pass_pipeline,
    scalar_type,
    specialize_and_lower,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.runtime.executors.statevector.reverse import (
    execute_torch_distributed_statevector_reverse,
)

pytestmark = pytest.mark.integration


def _seeded_source(seed: int) -> str:
    generator = random.Random(seed)
    branch = generator.choice((True, False))
    rotation = generator.choice((True, False))
    initial_wire = generator.randrange(3)
    selected_wire = generator.randrange(3)
    other_wire = (selected_wire + generator.randrange(1, 3)) % 3
    iterations = generator.randrange(1, 7)
    stride = generator.randrange(1, 3)
    return f"""
def seeded_program(theta, delta):
    angle = theta
    wire = {initial_wire}
    if {branch}:
        wire = {selected_wire}
        qp.H(wires=wire)
    else:
        wire = {other_wire}
        qp.X(wires=wire)
    for layer in range({iterations}):
        angle = angle + delta
        wire = (wire + {stride}) % 3
        if {rotation}:
            qp.RX(angle, wires=wire)
        else:
            qp.RY(angle, wires=wire)
    return qp.expval(qp.PauliZ(0))
"""


def _lower(
    source: str,
    theta: torch.Tensor,
    delta: torch.Tensor,
    *,
    optimize: bool,
) -> CircuitIR:
    program = capture_source(
        source,
        (scalar_type("float64"), scalar_type("float64")),
    )
    return specialize_and_lower(
        program,
        (theta, delta),
        circuit_dtype="complex128",
        optimize=optimize,
    ).bind()


def _instruction_structure(
    circuit: CircuitIR,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((item.name, item.wires) for item in circuit.instructions)


def _parameter_values(circuit: CircuitIR) -> torch.Tensor:
    return torch.stack(
        [
            item.params["theta"]
            for item in circuit.instructions
            if "theta" in item.params
        ]
    )


@pytest.mark.parametrize("seed", range(12))
def test_seeded_optimized_and_unoptimized_execution_and_vjp_agree(seed: int) -> None:
    source = _seeded_source(seed)
    optimized_theta = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    optimized_delta = torch.tensor(-0.03, dtype=torch.float64, requires_grad=True)
    baseline_theta = optimized_theta.detach().clone().requires_grad_(True)
    baseline_delta = optimized_delta.detach().clone().requires_grad_(True)

    program = capture_source(
        source,
        (scalar_type("float64"), scalar_type("float64")),
    )
    normalized = run_pass_pipeline(program)
    repeated = run_pass_pipeline(normalized.program)
    assert repeated.optimized_identity == normalized.optimized_identity
    assert all(record.changed is False for record in repeated.records)

    optimized = _lower(source, optimized_theta, optimized_delta, optimize=True)
    baseline = _lower(source, baseline_theta, baseline_delta, optimize=False)

    assert _instruction_structure(optimized) == _instruction_structure(baseline)
    torch.testing.assert_close(
        _parameter_values(optimized),
        _parameter_values(baseline),
        atol=0.0,
        rtol=0.0,
    )
    optimized_forward = fq.run(optimized).expectation()
    baseline_forward = fq.run(baseline).expectation()
    torch.testing.assert_close(
        optimized_forward,
        baseline_forward,
        atol=1e-12,
        rtol=1e-12,
    )

    optimized_vjp = execute_torch_distributed_statevector_reverse(
        optimized,
        observable_wires=(0,),
        device="cpu",
    )
    baseline_vjp = execute_torch_distributed_statevector_reverse(
        baseline,
        observable_wires=(0,),
        device="cpu",
    )
    optimized_vjp.backward()
    baseline_vjp.backward()

    torch.testing.assert_close(optimized_vjp.value, baseline_vjp.value)
    torch.testing.assert_close(optimized_theta.grad, baseline_theta.grad)
    torch.testing.assert_close(optimized_delta.grad, baseline_delta.grad)


def test_negative_runtime_loop_step_fails_equally_with_optimization_on_or_off() -> None:
    program = capture_source(
        """
def negative_step(step):
    for wire in range(2, 0, step):
        qp.X(wires=wire)
    return qp.expval(qp.PauliZ(0))
""",
        (INDEX,),
    )

    for optimize in (True, False):
        with pytest.raises(SpecializationError, match="index.value"):
            specialize_and_lower(program, (-1,), optimize=optimize)
