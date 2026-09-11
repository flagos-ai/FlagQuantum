import pytest
import torch

from flagquantum.simulation.mps.canonicalization import (
    absorb_left_canonical_transfer,
    absorb_right_canonical_transfer,
    deterministic_mps_qr,
    factor_left_canonical_site,
    factor_right_canonical_site,
    local_mixed_canonical_residual,
    mps_center_norms,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_deterministic_qr_fixes_diagonal_phase_and_reconstructs(dtype):
    matrix = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=dtype)
    first_q, first_r = deterministic_mps_qr(matrix)
    second_q, second_r = deterministic_mps_qr(matrix)
    torch.testing.assert_close(first_q, second_q)
    torch.testing.assert_close(first_r, second_r)
    torch.testing.assert_close(first_q @ first_r, matrix)
    diagonal = torch.diagonal(first_r)
    torch.testing.assert_close(
        diagonal.imag if diagonal.is_complex() else diagonal * 0,
        torch.zeros_like(diagonal.real),
    )
    assert torch.all(diagonal.real >= 0)


def test_left_and_right_canonical_steps_preserve_two_site_state():
    generator = torch.Generator().manual_seed(50)
    left = torch.randn((2, 2, 2, 3), dtype=torch.complex128, generator=generator)
    right = torch.randn((2, 3, 2, 2), dtype=torch.complex128, generator=generator)
    expected = torch.einsum("blsm,bmtr->blstr", left, right)

    canonical_left, right_transfer = factor_left_canonical_site(left)
    updated_right = absorb_left_canonical_transfer(right_transfer, right)
    torch.testing.assert_close(
        torch.einsum("blsm,bmtr->blstr", canonical_left, updated_right), expected
    )

    left_transfer, canonical_right = factor_right_canonical_site(right)
    updated_left = absorb_right_canonical_transfer(left, left_transfer)
    torch.testing.assert_close(
        torch.einsum("blsm,bmtr->blstr", updated_left, canonical_right), expected
    )


def test_canonical_residual_and_center_norms_are_local_numerics():
    center = torch.tensor([[[[1.0], [0.0]]], [[[0.0], [2.0]]]], dtype=torch.complex128)
    canonical = torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 2)

    residual = local_mixed_canonical_residual({0: canonical, 1: center}, center=1)

    torch.testing.assert_close(residual, torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(
        mps_center_norms(center), torch.tensor([1.0, 4.0], dtype=torch.float64)
    )
