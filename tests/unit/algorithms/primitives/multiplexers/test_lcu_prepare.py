"""Check signed coefficients and unused addresses in LCU preparation."""

import pytest
import torch

from flagquantum.algorithms.primitives.lcu_prepare import prepare_lcu

pytestmark = pytest.mark.unit


def test_prepare_lcu_pads_unused_addresses() -> None:
    prepare, alpha = prepare_lcu((1.0, -0.5, 0.5))
    assert alpha == 2.0
    torch.testing.assert_close(
        prepare.state().abs().square().flatten(),
        torch.tensor([0.5, 0.25, 0.25, 0.0], dtype=prepare.state().real.dtype),
    )


@pytest.mark.parametrize("coefficients", [(), (0.0,), (float("inf"),)])
def test_prepare_lcu_rejects_invalid_weights(coefficients: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="LCU"):
        prepare_lcu(coefficients)
