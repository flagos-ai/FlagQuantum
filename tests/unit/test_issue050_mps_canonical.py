import pytest
import torch

from flagquantum.runtime.backends.mps.canonicalization import _deterministic_qr

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_deterministic_qr_fixes_diagonal_phase_and_reconstructs(dtype):
    matrix = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=dtype)
    first_q, first_r = _deterministic_qr(matrix)
    second_q, second_r = _deterministic_qr(matrix)
    torch.testing.assert_close(first_q, second_q)
    torch.testing.assert_close(first_r, second_r)
    torch.testing.assert_close(first_q @ first_r, matrix)
    diagonal = torch.diagonal(first_r)
    torch.testing.assert_close(
        diagonal.imag if diagonal.is_complex() else diagonal * 0,
        torch.zeros_like(diagonal.real),
    )
    assert torch.all(diagonal.real >= 0)
