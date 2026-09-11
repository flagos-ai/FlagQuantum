from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.core.ir import ObservableNode
from flagquantum.runtime.capabilities import load_operator_profile
from flagquantum.runtime.executors.statevector.split_real_imag import (
    execute_split_real_imag_expectation,
    parameter_shift_split_real_imag_gradient,
    run_split_real_imag_training_conformance,
)
from flagquantum.runtime.operator_probes import (
    _preflight_split_real_imag_profile,
)

pytestmark = pytest.mark.unit


def _hamiltonian() -> Hamiltonian:
    return Hamiltonian(
        (
            pauli_term(0.7, "Z", (0,)),
            pauli_term(-0.4, "XX", (0, 1)),
            pauli_term(0.2, "Y", (1,)),
        )
    )


def _symbolic_circuit() -> Circuit:
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    return (
        Circuit(2).h(0).rx(0, theta=alpha).ry(1, theta=beta).rx(1, theta=alpha).cx(0, 1)
    )


def _complex128_reference(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    circuit = (
        Circuit(2, dtype=torch.complex128)
        .h(0)
        .rx(0, theta=alpha)
        .ry(1, theta=beta)
        .rx(1, theta=alpha)
        .cx(0, 1)
    )
    return _hamiltonian().expectation(circuit).sum()


def test_p1_hamiltonian_expectation_matches_complex128() -> None:
    bindings = {
        "alpha": torch.tensor(0.23, requires_grad=True),
        "beta": torch.tensor(-0.31, requires_grad=True),
    }
    result = execute_split_real_imag_expectation(
        _symbolic_circuit(),
        _hamiltonian(),
        parameter_bindings=bindings,
        preflight=False,
    )
    reference = _complex128_reference(
        torch.tensor(0.23, dtype=torch.float64),
        torch.tensor(-0.31, dtype=torch.float64),
    )

    torch.testing.assert_close(
        result.value.detach().cpu().to(torch.float64),
        reference.detach().cpu(),
        atol=2e-6,
        rtol=2e-6,
    )
    assert result.term_expectations.shape == (3,)
    assert result.value.dtype == torch.float32
    assert result.summary()["gradient_method"] == "parameter_shift"
    assert result.summary()["native_autograd_supported"] is False


def test_p1_parameter_shift_matches_complex128_autograd_with_shared_parameter() -> None:
    result = parameter_shift_split_real_imag_gradient(
        _symbolic_circuit(),
        _hamiltonian(),
        parameter_bindings={
            "alpha": torch.tensor(0.23, requires_grad=True),
            "beta": torch.tensor(-0.31, requires_grad=True),
        },
        preflight=False,
    )
    alpha = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(-0.31, dtype=torch.float64, requires_grad=True)
    reference = _complex128_reference(alpha, beta)
    reference.backward()
    expected = torch.stack((alpha.grad, beta.grad))

    assert result.parameter_order == ("alpha", "beta")
    assert result.shifted_evaluations == 6
    torch.testing.assert_close(
        result.gradient.detach().cpu().to(torch.float64),
        expected,
        atol=3e-6,
        rtol=3e-5,
    )


def test_p1_accepts_flagquantum_ir_pauli_observables() -> None:
    ir = replace(
        Circuit(2).h(0).cx(0, 1).to_ir(),
        observables=(
            ObservableNode(name="zz", wires=(0, 1), coefficient=0.5),
            ObservableNode(name="x", wires=(0,), coefficient=-0.25),
        ),
    )
    result = execute_split_real_imag_expectation(ir, preflight=False)
    assert result.value.item() == pytest.approx(0.5, abs=2e-6)


def test_p1_rejects_ambiguous_or_unsupported_parameter_shift() -> None:
    anonymous = torch.tensor(0.2, requires_grad=True)
    with pytest.raises(ValueError, match="named Parameter"):
        execute_split_real_imag_expectation(
            Circuit(1).rx(0, theta=anonymous),
            pauli_term(1.0, "Z", (0,)),
            preflight=False,
        )

    theta = Parameter("theta")
    with pytest.raises(NotImplementedError, match="ParameterExpression"):
        parameter_shift_split_real_imag_gradient(
            Circuit(1).rx(0, theta=2.0 * theta),
            pauli_term(1.0, "Z", (0,)),
            parameter_bindings={"theta": 0.2},
            preflight=False,
        )
    with pytest.raises(NotImplementedError, match="supports theta"):
        parameter_shift_split_real_imag_gradient(
            Circuit(1).u3(0, theta=theta, phi=0.1, lbd=-0.2),
            pauli_term(1.0, "Z", (0,)),
            parameter_bindings={"theta": 0.2},
            preflight=False,
        )


def test_p1_gradient_rejects_custom_input_and_batch() -> None:
    theta = Parameter("theta")
    observable = pauli_term(1.0, "Z", (0,))
    inputs = torch.tensor([[0.0, 1.0]], dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match="input only"):
        parameter_shift_split_real_imag_gradient(
            Circuit(1, inputs=inputs).rx(0, theta=theta),
            observable,
            parameter_bindings={"theta": 0.2},
            preflight=False,
        )
    with pytest.raises(NotImplementedError, match="batch size one"):
        parameter_shift_split_real_imag_gradient(
            Circuit(1, bsz=2).rx(0, theta=theta),
            observable,
            parameter_bindings={"theta": 0.2},
            preflight=False,
        )


def test_p1_profile_is_fp32_and_certifies_backward_probes() -> None:
    profile = load_operator_profile("split_real_imag_statevector_p1")
    assert {dtype for item in profile.requirements for dtype in item.dtypes} == {
        "float32"
    }
    assert any(item.backward for item in profile.requirements)
    report = _preflight_split_real_imag_profile(
        profile.name, device="cpu", provider="pytorch_cpu_test", refresh=True
    )
    assert report.supported
    assert len(report.evidence_ids) == len(profile.requirements)


def test_p1_training_conformance_against_complex128() -> None:
    report = run_split_real_imag_training_conformance("cpu", depths=(2, 8), seeds=(0,))
    report.require_accepted()
    assert report.operator_profile == "split_real_imag_statevector_p1"
    assert len(report.cases) == 2
