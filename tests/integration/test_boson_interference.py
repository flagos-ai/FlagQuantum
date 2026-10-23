"""Two-photon interference, uniform attenuation and explicit QND counting."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum.lindblad as fql
from examples.boson_interference import counting_instrument, interference_plan
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal

pytestmark = pytest.mark.integration


def attenuated_hom_state(cutoff: int, transmission: float) -> torch.Tensor:
    """Closed-form balanced HOM state with equal pure loss on both modes."""
    rho = torch.zeros((cutoff**2, cutoff**2), dtype=torch.complex128)
    rho[0, 0] = (1 - transmission) ** 2
    rho[1, 1] = rho[cutoff, cutoff] = transmission * (1 - transmission)
    for row in (2, 2 * cutoff):
        for column in (2, 2 * cutoff):
            rho[row, column] = transmission**2 / 2
    return rho


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_interference_loss_and_cutoff_match_closed_form(rate: float) -> None:
    transmission = math.exp(-rate * math.pi / (4 * 0.7))
    densities: list[torch.Tensor] = []
    for cutoff in (3, 4):
        plan = interference_plan(rate, cutoff=cutoff)
        encoded = plan.to_json()
        actual = fql.run(plan)
        replay = fql.run(fql.LindbladPlan.from_json(encoded))
        assert actual.density_matrices is not None
        assert replay.density_matrices is not None
        rho = actual.density_matrices[-1]
        expected = attenuated_hom_state(cutoff, transmission)
        torch.testing.assert_close(rho, expected, atol=1e-10, rtol=0)
        torch.testing.assert_close(
            actual.populations[-1], expected.diag().real, atol=1e-10, rtol=0
        )
        assert torch.equal(actual.density_matrices, replay.density_matrices)
        assert encoded == plan.to_json()
        densities.append(rho)
    embedding = torch.tensor(
        [4 * left + right for left in range(3) for right in range(3)]
    )
    embedded = densities[1].index_select(0, embedding).index_select(1, embedding)
    torch.testing.assert_close(densities[0], embedded, atol=1e-10, rtol=0)


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_number_counts_unconditional_loss_and_measurement_backaction(
    rate: float,
) -> None:
    evolution = fql.run(interference_plan(rate))
    assert evolution.density_matrices is not None
    rho = evolution.density_matrices[-1]
    transmission = math.exp(-rate * math.pi / (4 * 0.7))
    expected = attenuated_hom_state(3, transmission).diag().real
    instrument = counting_instrument()
    rng = torch.get_rng_state()
    observed = sample_terminal(rho, instrument, shots=4096, seed=7)
    repeated = sample_terminal(rho, instrument, shots=4096, seed=7)
    assert isinstance(observed.value, torch.Tensor)
    assert isinstance(repeated.value, torch.Tensor)
    assert torch.equal(observed.value, repeated.value)
    assert torch.equal(rng, torch.get_rng_state())
    assert observed.metadata["outcome_labels"] == [
        f"{left}/{right}" for left in range(3) for right in range(3)
    ]
    weights = torch.tensor(observed.metadata["probabilities"], dtype=torch.float64)
    torch.testing.assert_close(weights, expected, atol=1e-10, rtol=0)
    counts = torch.bincount(observed.value, minlength=9)
    assert counts.sum().item() == 4096
    active = expected > 0
    frequencies = counts[active].to(torch.float64) / 4096
    standard_errors = torch.sqrt(expected[active] * (1 - expected[active]) / 4096)
    assert bool(((frequencies - expected[active]).abs() < 6 * standard_errors).all())
    assert bool((counts[~active] == 0).all())
    # Retain vacuum and one-photon outcomes; do not silently postselect survivors.
    torch.testing.assert_close(
        weights[[2, 4, 6]].sum(),
        torch.tensor(transmission**2, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    unread = torch.stack(
        [SuperOperator.from_kraus(branch).apply(rho) for branch in instrument.values()]
    ).sum(dim=0)
    torch.testing.assert_close(
        unread, torch.diag(expected).to(rho.dtype), atol=1e-10, rtol=0
    )
    assert rho[2, 6].abs().item() == pytest.approx(transmission**2 / 2, abs=1e-10)
    assert unread[2, 6].abs().item() == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("cutoff", [True, 2, 5])
def test_reference_refuses_cutoffs_outside_its_declared_comparison(cutoff: int) -> None:
    with pytest.raises(ValueError, match="three or four"):
        interference_plan(0.2, cutoff=cutoff)
