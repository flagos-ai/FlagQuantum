import pytest
import torch

pytest.importorskip("triton")

from flagquantum.simulation.triton_kernels.two_qubit_pauli_tangent import (
    _reference,
    repeated_rxx_ryy_rzz_tangents,
)


def reference_tangents(state, angles):
    real = torch.autograd.functional.jacobian(
        lambda value: _reference(state, value).real, angles, vectorize=True
    )
    imag = torch.autograd.functional.jacobian(
        lambda value: _reference(state, value).imag, angles, vectorize=True
    )
    jacobian = torch.complex(real, imag)
    indices = torch.arange(int(state.shape[0]), device=state.device)
    return (
        jacobian[indices, :, :, indices]
        .movedim((-2, -1), (0, 1))
        .reshape(3 * angles.shape[1], *state.shape)
    )


def test_two_qubit_pauli_tangent_cpu_fallback_matches_reference():
    torch.manual_seed(3)
    state = torch.randn(2, 5, 4, dtype=torch.complex64)
    angles = torch.randn(2, 3, 3)
    torch.testing.assert_close(
        repeated_rxx_ryy_rzz_tangents(state, angles),
        reference_tangents(state, angles),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 4, 12])
def test_two_qubit_pauli_tangent_cuda_matches_reference(depth):
    torch.manual_seed(7)
    state = torch.randn(2, 257, 4, dtype=torch.complex64)
    angles = torch.randn(2, depth, 3)
    expected = reference_tangents(state, angles)
    actual = repeated_rxx_ryy_rzz_tangents(state.cuda(), angles.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
