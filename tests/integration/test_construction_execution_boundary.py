"""Real CPU probes of the existing boundaries required by CS-01.

These execute existing circuits, not synthesized arithmetic. The construction
profile has no gate lowering yet; these probes establish its future IR boundary.
"""

import cmath
import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler import final_layout
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.runtime.execution_plan_contract import (
    plan_execution_program,
    plan_program,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def options() -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def test_result_is_bound_to_normalized_execution_not_source(
    options: fq.ExecutionOptions,
) -> None:
    source = fq.Circuit(2).x(0).x(0).h(1).to_ir()
    plan = fq.plan(source, options=options)
    executable = plan_execution_program(plan)
    assert source.content_hash != plan.program_fingerprint
    assert plan_program(plan).content_hash == plan.program_fingerprint
    assert executable.content_hash == plan.program_fingerprint
    result = fq.run(plan)
    assert result.plan is plan
    assert result.plan.identity == plan.identity
    expected = torch.tensor([1, 1, 0, 0], dtype=torch.complex128) / math.sqrt(2)
    torch.testing.assert_close(
        result.statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )
    other = fq.plan(fq.Circuit(2).h(0), options=options)
    assert other.program_fingerprint != plan.program_fingerprint
    assert other.identity != result.plan.identity


def test_parameter_values_participate_in_executable_identity(
    options: fq.ExecutionOptions,
) -> None:
    first = fq.plan(fq.Circuit(1).ry(0, 0.3), options=options)
    second = fq.plan(fq.Circuit(1).ry(0, 0.7), options=options)
    assert first.program_fingerprint != second.program_fingerprint
    assert first.identity != second.identity
    assert not torch.allclose(fq.run(first).statevector(), fq.run(second).statevector())


def test_existing_routing_restores_named_logical_order(
    options: fq.ExecutionOptions,
) -> None:
    source = fq.Circuit(3).x(0).cx(0, 2)
    routed = compile_program(source, coupling_map=((0, 1), (1, 2)), optimize=False)
    assert any(item.name == "swap" for item in routed.instructions)
    assert final_layout(routed).logical_to_physical == (0, 1, 2)
    result = fq.run(fq.plan(routed, options=options))
    reference = torch.zeros(8, dtype=torch.complex128)
    reference[5] = 1  # Wire 0 is the MSB: logical |101>.
    torch.testing.assert_close(
        result.statevector().reshape(-1), reference, atol=1e-12, rtol=0
    )


def test_full_core_block_preserves_angles_and_controlled_global_phase(
    options: fq.ExecutionOptions,
) -> None:
    theta, phi, lbd, gamma = 0.7, -0.2, 0.4, 0.3
    block = CircuitIR(
        n_wires=1,
        dtype="complex128",
        instructions=(
            Instruction("u3", (0,), {"theta": theta, "phi": phi, "lbd": lbd}),
            Instruction(
                "unitary",
                (0,),
                matrix=torch.eye(2, dtype=torch.complex128) * cmath.exp(1j * gamma),
            ),
        ),
    )
    restored = CircuitIR.from_json(block.to_json())
    assert restored.instructions[0].params == block.instructions[0].params
    controlled = fq.Circuit.from_ir(restored).control(1, 1)
    circuit = fq.Circuit(2).h(0).h(1).compose(controlled)
    result = fq.run(circuit, options=options)
    u = torch.tensor(
        [
            [math.cos(theta / 2), -cmath.exp(1j * lbd) * math.sin(theta / 2)],
            [
                cmath.exp(1j * phi) * math.sin(theta / 2),
                cmath.exp(1j * (phi + lbd)) * math.cos(theta / 2),
            ],
        ],
        dtype=torch.complex128,
    )
    active = cmath.exp(1j * gamma) * (u @ torch.ones(2, dtype=torch.complex128)) / 2
    reference = torch.tensor([0.5, active[0], 0.5, active[1]], dtype=torch.complex128)
    torch.testing.assert_close(
        result.statevector().reshape(-1), reference, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("wire,integer", [(2, 1), (0, 4)])
def test_non_palindromic_integer_bit_order_boundary(
    options: fq.ExecutionOptions, wire: int, integer: int
) -> None:
    # Physical-order sentinel for the future adapter, not a Qmod import test.
    result = fq.run(fq.Circuit(3).x(wire), options=options)
    assert int(result.statevector().reshape(-1).abs().argmax()) == integer


def test_omitting_an_entangled_output_does_not_clean_or_reclaim_it(
    options: fq.ExecutionOptions,
) -> None:
    # Establish the subsystem semantics behind the future drop contract using
    # existing output selection. No drop operation is implemented here.
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(circuit, options=options, outputs=fq.density_matrix((0,)))
    reduced = result.density_matrix.reshape(2, 2)
    torch.testing.assert_close(
        reduced, torch.eye(2, dtype=torch.complex128) / 2, atol=1e-12, rtol=0
    )
    assert result.plan.analysis.n_wires == 2
    assert float(torch.trace(reduced @ reduced).real) == pytest.approx(0.5)
