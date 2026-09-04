import pytest
import torch

from flagquantum.simulation.mps_reverse import mps_vjp, project_mps_adjoint

pytestmark = pytest.mark.unit


def test_mps_vjp_projects_truncated_bonds_and_returns_input_parameter_grads() -> None:
    value = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    parameter = torch.tensor(3.0, requires_grad=True)
    output = value * parameter
    stale_adjoint = torch.tensor([[[2.0, 4.0, 8.0]]])

    value_grad, parameter_grad = mps_vjp(
        (output,),
        (value,),
        (parameter,),
        (stale_adjoint,),
    )

    torch.testing.assert_close(value_grad, torch.tensor([[[6.0, 12.0]]]))
    torch.testing.assert_close(parameter_grad, torch.tensor(10.0))


def test_mps_adjoint_projection_rejects_rank_or_batch_mismatch() -> None:
    target = torch.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError, match="rank/batch mismatch"):
        project_mps_adjoint(torch.zeros((1, 1, 2, 1)), target)
