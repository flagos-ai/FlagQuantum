from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.bindings import BindingTable
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.testing.differential import (
    execute_differential,
    lower_sealed_for_differential,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype", "atol"),
    [
        ("complex64", torch.float32, 1e-6),
        ("complex128", torch.float64, 1e-12),
    ],
)
def test_trainable_tensor_forward_and_gradient_parity(
    complex_dtype: str, real_dtype: torch.dtype, atol: float
) -> None:
    theta = torch.tensor(0.37, dtype=real_dtype, requires_grad=True)
    source = fq.CircuitIR(
        1,
        (fq.Instruction("rx", (0,), {"theta": theta}),),
        dtype=complex_dtype,
        measurements=(fq.MeasurementNode("expectation_z", (0,)),),
    )

    def differentiable_expectation(ir: fq.CircuitIR) -> torch.Tensor:
        circuit = fq.Circuit.from_ir(ir, dtype=getattr(torch, complex_dtype))
        return circuit.expectation_z(0).sum()

    execution = execute_differential(source, differentiable_expectation)
    legacy_loss = execution.legacy_result
    candidate_loss = execution.candidate_result
    legacy_gradient = torch.autograd.grad(legacy_loss, theta, retain_graph=True)[0]
    candidate_gradient = torch.autograd.grad(candidate_loss, theta)[0]

    assert execution.report.ok
    assert execution.report.binding_identity_preserved is True
    torch.testing.assert_close(legacy_loss, candidate_loss, atol=atol, rtol=0)
    torch.testing.assert_close(legacy_gradient, candidate_gradient, atol=atol, rtol=0)
    torch.testing.assert_close(
        legacy_gradient, -torch.sin(theta.detach()), atol=atol, rtol=0
    )


def test_parameter_and_expression_identity_survives_test_lowering() -> None:
    theta = fq.Parameter("theta")
    source = fq.CircuitIR(
        1,
        (
            fq.Instruction("rx", (0,), {"theta": theta}),
            fq.Instruction("rz", (0,), {"theta": theta * 0.5}),
        ),
    )
    artifact = seal_circuit_ir_round_trip(source).artifact

    lowered = lower_sealed_for_differential(artifact)

    assert lowered.ok
    first = lowered.circuit_ir.instructions[0].params["theta"]
    second = lowered.circuit_ir.instructions[1].params["theta"]
    assert isinstance(first, fq.Parameter)
    assert isinstance(second, fq.ParameterExpression)
    assert first.name == "theta"
    assert second.op == "mul"
    assert second.args[0].name == "theta"


def test_binding_shape_or_dtype_drift_fails_closed() -> None:
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    source = fq.CircuitIR(
        1,
        (fq.Instruction("rx", (0,), {"theta": theta}),),
        dtype="complex128",
    )
    artifact = seal_circuit_ir_round_trip(source).artifact
    reference = artifact.imported.bindings.references[0]
    wrong_reference = replace(reference, dtype="float32")
    tampered_import = replace(
        artifact.imported,
        bindings=BindingTable(((wrong_reference, theta),)),
    )
    artifact = replace(artifact, imported=tampered_import)

    lowered = lower_sealed_for_differential(artifact)

    assert not lowered.ok
    assert lowered.circuit_ir is None
    assert "dtype changed" in lowered.diagnostics[0].message
