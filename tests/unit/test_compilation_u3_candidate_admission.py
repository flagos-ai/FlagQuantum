"""Exercise target admission before absolute-phase U3 candidate construction."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.optimization import one_qubit_synthesis
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit


def _product(instructions: tuple[Instruction, ...]) -> torch.Tensor:
    result = torch.eye(2, dtype=torch.complex128)
    for instruction in instructions:
        matrix = gate_matrix(
            instruction, bsz=1, device="cpu", dtype=torch.complex128
        ).reshape(2, 2)
        result = matrix @ result
    return result


@pytest.mark.parametrize("opcode", ["rz", "sx", "sxdg"])
def test_non_u3_target_does_not_construct_u3_phase_correction(opcode: str) -> None:
    params = {"theta": 0.37} if opcode == "rz" else {}
    source = Instruction(opcode, (0,), params, metadata={"source": "rotation"})
    with patch.object(one_qubit_synthesis, "Instruction", wraps=Instruction) as build:
        converted = convert_basis(CircuitIR(1, (source,)), gates=("phase", "rx", "cz"))
    u3_constructions = [call for call in build.call_args_list if call.args[0] == "u3"]
    assert len(u3_constructions) == 1
    assert all(
        gate.name in {"phase", "rx", "cz"} for gate in converted.program.instructions
    )
    assert all(
        dict(gate.metadata) == dict(source.metadata)
        for gate in converted.program.instructions
    )
    actual, expected = _product(converted.program.instructions), _product((source,))
    overlap = torch.vdot(actual.flatten(), expected.flatten()) / 2
    torch.testing.assert_close(expected, overlap * actual, atol=1e-12, rtol=0)


@pytest.mark.parametrize("rejected", ["primary", "x_correction", "phase_correction"])
def test_u3_candidate_requires_every_parameter_constrained_leaf(rejected: str) -> None:
    source = Instruction("rz", (0,), {"theta": 0.37})

    def can_run(gate: Instruction) -> bool:
        if gate.name != "u3":
            return False
        theta, angle = gate.params["theta"], gate.params["lbd"]
        if rejected == "primary":
            return angle != 0.37
        if rejected == "x_correction":
            return theta == 0.0
        return angle != -0.185

    assert translate(source, can_run=can_run) is None


def test_u3_target_keeps_absolute_phase_and_trainable_gradient() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    source = Instruction("rz", (0,), {"theta": theta}, metadata={"tag": 4})
    leaves = translate(source, can_run=lambda gate: gate.name == "u3")
    assert leaves is not None and len(leaves) == 5
    assert all(dict(gate.metadata) == dict(source.metadata) for gate in leaves)
    actual, expected = _product(leaves), _product((source,))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    # Absolute complex amplitudes detect losing the scalar correction.
    expected_loss = expected.real.sum() + 0.3 * expected.imag.sum()
    actual_loss = actual.real.sum() + 0.3 * actual.imag.sum()
    expected_gradient = torch.autograd.grad(expected_loss, theta, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual_loss, theta)[0]
    assert expected_gradient.abs() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=0)


def test_symbolic_u3_target_binds_source_and_scalar_correction_together() -> None:
    source = fq.Circuit(1).rz(0, 2 * Parameter("theta") + 0.1)
    converted = convert_basis(source, gates=("u3",))
    output = fq.Circuit.from_ir(converted.program)
    assert len(output.to_ir().instructions) == 5
    for value in (0.0, 0.37, -0.61):
        expected = source.bind_parameters({"theta": value})
        actual = output.bind_parameters({"theta": value})
        torch.testing.assert_close(
            _product(actual.to_ir().instructions),
            _product(expected.to_ir().instructions),
            atol=1e-12,
            rtol=0,
        )
