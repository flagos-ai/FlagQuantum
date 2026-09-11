"""Validate the optimizer's single-tensor autograd boundary."""

import pytest
import torch

from flagquantum.algorithms.optimization import _tensor_jacobian

pytestmark = pytest.mark.unit


def test_tensor_jacobian_matches_analytic_derivative() -> None:
    inputs = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)

    actual = _tensor_jacobian(lambda value: value.square(), inputs)

    torch.testing.assert_close(actual, torch.diag(2 * inputs))
    assert not actual.requires_grad
    assert inputs.grad is None


@pytest.mark.parametrize("result", [None, (torch.tensor(1.0),)])
def test_tensor_jacobian_rejects_nontensor_results(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    def malformed_jacobian(*args: object, **kwargs: object) -> object:
        return result

    monkeypatch.setattr(torch.autograd.functional, "jacobian", malformed_jacobian)

    with pytest.raises(TypeError, match="Single-tensor Jacobian must return a tensor"):
        _tensor_jacobian(lambda value: value.square(), torch.tensor([0.2]))
