"""Real-device correctness for the FlagTree-owned TLE statevector kernel."""

from __future__ import annotations

import pytest
import torch

from flagquantum.kernels.flagtree import apply_complex64_local_1q_tle
from flagquantum.kernels.provenance import triton_compiler_provenance

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def _reference_apply(
    state: torch.Tensor,
    matrix: torch.Tensor,
    bit_position: int,
) -> torch.Tensor:
    batch, size = state.shape
    high = size >> (bit_position + 1)
    paired = state.reshape(batch, high, 2, 1 << bit_position)
    return torch.einsum("ij,bhjw->bhiw", matrix, paired).reshape_as(state)


@pytest.mark.parametrize("bit_position", [0, 5, 9])
def test_flagtree_tle_local_1q_matches_pytorch(bit_position: int) -> None:
    distribution, version, integration_path, status = triton_compiler_provenance()
    assert (distribution, integration_path, status) == (
        "flagtree",
        "flagtree",
        "resolved",
    )
    assert version == "0.7.0"

    generator = torch.Generator(device="cuda").manual_seed(261005 + bit_position)
    state = torch.randn(
        2,
        1 << 10,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    matrix = torch.randn(
        2,
        2,
        dtype=torch.complex64,
        device="cuda",
        generator=generator,
    )
    expected = _reference_apply(state, matrix, bit_position)
    actual = apply_complex64_local_1q_tle(
        state,
        matrix,
        bit_position=bit_position,
    )

    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
