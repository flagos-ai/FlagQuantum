"""Batched MPS truncation must retain the subspace needed by every sample."""

import pytest
import torch

from flagquantum.simulation.mps.factorization import (
    _split_pair_matrix,
    _split_pair_matrix_bucket,
)
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_statevector_conversion_preserves_batched_entanglement() -> None:
    states = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]],
        dtype=torch.complex128,
    )
    result = MPSState.from_statevector(states, 2, config=MPSConfig(cutoff=0.05))
    torch.testing.assert_close(result.to_statevector(), states)
    assert result.bond_dims == (2,)
    assert result.truncation_records[0].discarded_weight == 0.0


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("max_bond", [None, 2])
def test_shared_rank_preserves_each_samples_cutoff(
    bucketed: bool, requires_grad: bool, max_bond: int | None
) -> None:
    spectra = torch.tensor(
        [[0.9, 0.01, 0.005, 0.001], [0.8, 0.5, 0.3, 0.002]],
        dtype=torch.float64,
    )
    matrix = torch.diag_embed(spectra).requires_grad_(requires_grad)
    config = MPSConfig(cutoff=0.05, max_bond=max_bond)
    if bucketed:
        # Each bond chooses its own rank, while samples share a rank per bond.
        low_rank_bond = matrix[:1].expand_as(matrix)
        result, other_result = _split_pair_matrix_bucket(
            torch.stack((matrix, low_rank_bond)),
            left_dim=2,
            right_dim=2,
            config=config,
        )
        assert other_result[2]["rank"] == 1
    else:
        result = _split_pair_matrix(matrix, left_dim=2, right_dim=2, config=config)

    left, right, diagnostics = result
    rank = 3 if max_bond is None else max_bond
    assert diagnostics["rank"] == rank
    reconstructed = left.reshape(2, 4, rank) @ right.reshape(2, rank, 4)
    expected = matrix.detach().clone()
    expected[:, rank:, :] = 0
    torch.testing.assert_close(reconstructed, expected)
    discarded = spectra[:, rank:].square().sum(dim=-1).max().item()
    assert diagnostics["discarded_weight"] == pytest.approx(discarded)
    if requires_grad:
        reconstructed.square().sum().backward()
        torch.testing.assert_close(matrix.grad, 2 * expected)
