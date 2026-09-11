import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.simulation.mps.static import (
    CX,
    StaticMPSProgram,
    build_bucketed_static_vqe_loss,
    build_static_vqe_loss,
    build_static_vqe_loss_real_imag,
    compile_static_vqe_loss,
    run_static_brickwork,
    ry_matrix,
    rz_matrix,
)
from flagquantum.simulation.statevector.operations import _apply_matrix


def test_static_bond_profile_and_zero_state():
    program = StaticMPSProgram.compile(8, 16)
    assert program.bond_dims == (1, 2, 4, 8, 16, 8, 4, 2, 1)
    state = program.to_statevector(program.zero())
    assert state.shape == (1, 2**8)
    assert state[0, 0] == 1


def test_ir_inferred_brickwork_profile_is_depth_bounded():
    circuit = Circuit(128)
    for layer in range(8):
        for wire in range(layer % 2, 127, 2):
            circuit.cx(wire, wire + 1)
    program = StaticMPSProgram.from_ir(circuit.to_ir(), max_bond=16)
    assert max(program.bond_dims) == 16
    assert max(program.crossing_counts) == 4
    assert program.bond_dims[64] == 16
    with pytest.raises(ValueError, match="requires exact bond 16"):
        StaticMPSProgram.from_ir(circuit.to_ir(), max_bond=8)


def test_static_brickwork_matches_dense_forward_and_gradient():
    torch.manual_seed(7)
    parameters = (0.1 * torch.randn(2, 6, 2)).requires_grad_(True)
    reference_parameters = parameters.detach().clone().requires_grad_(True)
    program = StaticMPSProgram.compile(6, 8)
    tensors = run_static_brickwork(program, parameters)
    actual = program.to_statevector(tensors)

    reference = torch.zeros(1, 2**6, dtype=torch.complex64)
    reference[:, 0] = 1
    for layer in range(2):
        for wire in range(6):
            reference = _apply_matrix(
                reference, ry_matrix(reference_parameters[layer, wire, 0]), (wire,), 6
            )
            reference = _apply_matrix(
                reference, rz_matrix(reference_parameters[layer, wire, 1]), (wire,), 6
            )
        for wire in range(layer % 2, 5, 2):
            reference = _apply_matrix(reference, CX, (wire, wire + 1), 6)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    actual_loss = program.expectation_z_zz_chain(tensors).sum()
    probabilities = reference.abs().square()
    indices = torch.arange(2**6)
    reference_loss = torch.zeros(())
    for wire in range(6):
        z = 1 - 2 * ((indices >> (5 - wire)) & 1)
        reference_loss = reference_loss + 0.1 * torch.sum(probabilities * z)
    for wire in range(5):
        zl = 1 - 2 * ((indices >> (5 - wire)) & 1)
        zr = 1 - 2 * ((indices >> (4 - wire)) & 1)
        reference_loss = reference_loss - torch.sum(probabilities * zl * zr)
    torch.testing.assert_close(actual_loss, reference_loss, atol=2e-5, rtol=2e-5)
    actual_gradient = torch.autograd.grad(actual_loss, parameters)[0]
    reference_gradient = torch.autograd.grad(reference_loss, reference_parameters)[0]
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=2e-4, rtol=2e-4
    )


def test_truncated_static_gradient_is_finite():
    torch.manual_seed(11)
    parameters = (0.1 * torch.randn(4, 8, 2)).requires_grad_(True)
    program = StaticMPSProgram.compile(8, 2)
    loss = build_static_vqe_loss(program, 4)(parameters)
    gradient = torch.autograd.grad(loss, parameters)[0]
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(gradient))


def test_real_imag_loss_and_gradient_match_complex_static():
    torch.manual_seed(13)
    parameters = (0.1 * torch.randn(2, 6, 2)).requires_grad_(True)
    real_imag_parameters = parameters.detach().clone().requires_grad_(True)
    program = StaticMPSProgram.compile(6, 8)
    complex_loss = build_static_vqe_loss(program, 2)(parameters)
    real_imag_loss = build_static_vqe_loss_real_imag(program, 2)(real_imag_parameters)
    complex_gradient = torch.autograd.grad(complex_loss, parameters)[0]
    real_imag_gradient = torch.autograd.grad(real_imag_loss, real_imag_parameters)[0]
    torch.testing.assert_close(real_imag_loss, complex_loss, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        real_imag_gradient, complex_gradient, atol=2e-4, rtol=2e-4
    )


def test_static_compile_cache_with_eager_backend():
    program = StaticMPSProgram.compile(4, 4)
    first = compile_static_vqe_loss(program, 1, backend="eager")
    second = compile_static_vqe_loss(program, 1, backend="eager")
    assert first is second
    parameters = torch.zeros(1, 4, 2, requires_grad=True)
    loss = first(parameters)
    assert torch.isfinite(torch.autograd.grad(loss, parameters)[0]).all()


def test_bucketed_scheduler_matches_unbucketed_loss_and_gradient():
    torch.manual_seed(17)
    parameters = (0.1 * torch.randn(2, 6, 2)).requires_grad_(True)
    bucketed_parameters = parameters.detach().clone().requires_grad_(True)
    program = StaticMPSProgram.compile(6, 8)
    expected = build_static_vqe_loss_real_imag(program, 2)(parameters)
    actual = build_bucketed_static_vqe_loss(program, 2, backend="eager")(
        bucketed_parameters
    )
    expected_gradient = torch.autograd.grad(expected, parameters)[0]
    actual_gradient = torch.autograd.grad(actual, bucketed_parameters)[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-4, rtol=2e-4)
