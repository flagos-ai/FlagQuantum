"""Local slice-reduction contracts, without distributed capacity claims."""

import pytest
import torch

from flagquantum.runtime.executors.tensor_network import execution

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("requires_grad", [False, True])
def test_slice_reduction_preserves_input_and_local_gradient(
    monkeypatch: pytest.MonkeyPatch, requires_grad: bool
) -> None:
    calls = 0

    def add_remote_contribution(tensor: torch.Tensor, *, op: object) -> None:
        nonlocal calls
        calls += 1
        assert op == torch.distributed.ReduceOp.SUM
        tensor.add_(torch.tensor([4.0, 5.0], dtype=tensor.dtype))

    monkeypatch.setattr(execution.dist, "all_reduce", add_remote_contribution)
    local = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=requires_grad)
    result = execution._all_reduce_sum_autograd(local)
    torch.testing.assert_close(result, torch.tensor([5.0, 7.0], dtype=local.dtype))
    torch.testing.assert_close(local, torch.tensor([1.0, 2.0], dtype=local.dtype))
    assert result.requires_grad == requires_grad
    if requires_grad:
        weights = torch.tensor([2.0, 3.0], dtype=local.dtype)
        (result * weights).sum().backward()
        torch.testing.assert_close(local.grad, weights)
    assert calls == 1


def test_slice_reduction_rejects_non_tensor_framework_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution._DistributedAllReduceSum, "apply", lambda *args: None)
    with pytest.raises(TypeError, match="must return a tensor"):
        execution._all_reduce_sum_autograd(torch.ones(2, requires_grad=True))
