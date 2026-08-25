from __future__ import annotations

import math

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import pauli_term
from flagquantum.runtime.backends.statevector.split_real_imag_autograd import (
    split_real_imag_device_double_single_autograd_expectation,
    split_real_imag_p5_autograd_bridge_summary,
)

pytestmark = pytest.mark.unit


def _loss(theta: torch.Tensor) -> torch.Tensor:
    parameter = Parameter("theta")
    return split_real_imag_device_double_single_autograd_expectation(
        Circuit(1).ry(0, theta=parameter),
        pauli_term(1.0, "Z", (0,)),
        parameter_bindings={"theta": theta},
        preflight=False,
    )


def test_p5_autograd_bridge_delivers_expected_float32_gradient() -> None:
    theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    loss = _loss(theta)
    loss.backward()
    assert loss.dtype == torch.float32
    assert theta.grad is not None
    assert theta.grad.dtype == torch.float32
    assert abs(loss.item() - math.cos(0.23)) < 1e-6
    assert abs(theta.grad.item() + math.sin(0.23)) < 1e-6


def test_p5_autograd_bridge_preserves_named_parameter_order() -> None:
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    circuit = Circuit(1).ry(0, theta=beta).rz(0, theta=alpha)
    bindings = {
        "beta": torch.tensor(0.31, dtype=torch.float32, requires_grad=True),
        "alpha": torch.tensor(-0.17, dtype=torch.float32, requires_grad=True),
    }
    loss = split_real_imag_device_double_single_autograd_expectation(
        circuit,
        pauli_term(1.0, "X", (0,)),
        parameter_bindings=bindings,
        preflight=False,
    )
    loss.backward()
    assert bindings["alpha"].grad is not None
    assert bindings["beta"].grad is not None


def test_p5_autograd_bridge_passes_bounded_float32_gradcheck() -> None:
    theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    with pytest.warns(UserWarning, match="not a double precision"):
        assert torch.autograd.gradcheck(
            _loss,
            (theta,),
            eps=1e-3,
            atol=1e-4,
            rtol=1e-3,
        )


def test_p5_autograd_bridge_rejects_invalid_precision_and_binding_scope() -> None:
    parameter = Parameter("theta")
    circuit = Circuit(1).ry(0, theta=parameter)
    observable = pauli_term(1.0, "Z", (0,))
    with pytest.raises(TypeError, match="torch.float32"):
        split_real_imag_device_double_single_autograd_expectation(
            circuit,
            observable,
            parameter_bindings={
                "theta": torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
            },
            preflight=False,
        )
    with pytest.raises(ValueError, match="require gradients"):
        split_real_imag_device_double_single_autograd_expectation(
            circuit,
            observable,
            parameter_bindings={"theta": torch.tensor(0.2, dtype=torch.float32)},
            preflight=False,
        )
    with pytest.raises(ValueError, match="exactly match"):
        split_real_imag_device_double_single_autograd_expectation(
            circuit,
            observable,
            parameter_bindings={
                "other": torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
            },
            preflight=False,
        )


def test_p5_autograd_bridge_rejects_higher_order_gradients() -> None:
    theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(_loss(theta), theta, create_graph=True)


def test_p5_autograd_metadata_keeps_precision_and_claim_boundaries_explicit() -> None:
    summary = split_real_imag_p5_autograd_bridge_summary()
    assert summary["delivered_tensor_grad_precision"] == "float32_boundary"
    assert summary["internal_gradient_representation"] == "double_single_high_low"
    assert summary["end_to_end_double_single_gradient_claim_allowed"] is False
    assert summary["optimizer_available"] is False
    assert summary["native_cuda_evidence"] is False
    assert summary["torch_fl_flagos_evidence"] is False
    assert summary["flagcx_collectives_validated"] is False
    assert summary["convergence_certification"] is False
