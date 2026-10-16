"""Real-device correctness for FlagTree-owned TLE adjoint kernels."""

from __future__ import annotations

import pytest
import torch

from flagquantum.kernels.flagtree import (
    fused_complex64_sharded_1q_vjp_adjoint_tle,
)
from flagquantum.kernels.provenance import triton_compiler_provenance

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def _require_flagtree_070() -> None:
    distribution, version, integration_path, status = triton_compiler_provenance()
    assert (distribution, version, integration_path, status) == (
        "flagtree",
        "0.7.0",
        "flagtree",
        "resolved",
    )


@pytest.mark.parametrize("rank_basis", [0, 1])
def test_flagtree_tle_sharded_vjp_matches_global_reference(rank_basis: int) -> None:
    _require_flagtree_070()
    generator = torch.Generator(device="cuda").manual_seed(261007 + rank_basis)
    before = torch.randn(
        2,
        2,
        4096,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    adjoint = torch.randn_like(before)
    matrix = torch.randn(
        2,
        2,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    derivative = torch.randn(
        2,
        2,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )

    next_adjoint, gradient = fused_complex64_sharded_1q_vjp_adjoint_tle(
        before[:, rank_basis].contiguous(),
        before[:, 1 - rank_basis].contiguous(),
        adjoint[:, rank_basis].contiguous(),
        adjoint[:, 1 - rank_basis].contiguous(),
        matrix,
        derivative,
        rank_basis=rank_basis,
    )
    expected_adjoint = torch.einsum("ij,bjk->bik", matrix.mH, adjoint)
    derivative_state = torch.einsum("ij,bjk->bik", derivative, before)
    expected_gradient = torch.real(
        torch.sum(torch.conj(adjoint[:, rank_basis]) * derivative_state[:, rank_basis])
    )

    torch.testing.assert_close(
        next_adjoint,
        expected_adjoint[:, rank_basis],
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        gradient,
        expected_gradient,
        atol=3e-3,
        rtol=3e-5,
    )
