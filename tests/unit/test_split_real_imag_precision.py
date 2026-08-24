from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.core.numerics import AccuracyRequirementContract, PrecisionPlanContract
from flagquantum.runtime.backends.statevector.split_real_imag import (
    execute_split_real_imag_expectation,
    parameter_shift_split_real_imag_gradient,
)
from flagquantum.runtime.backends.statevector.split_real_imag_precision import (
    execute_split_real_imag_precision_expectation,
    parameter_shift_split_real_imag_precision_gradient,
    run_split_real_imag_precision_conformance,
    split_real_imag_p2_precision_plan,
)
from flagquantum.runtime.capabilities import load_operator_profile
from flagquantum.runtime.operator_probes import (
    preflight_split_real_imag_statevector_p2,
)

pytestmark = pytest.mark.unit


def _cancellation_hamiltonian() -> Hamiltonian:
    return Hamiltonian(
        tuple(pauli_term(coefficient, "Z", (0,)) for coefficient in (1e8, 1.0, -1e8))
    )


def test_p2_selective_reduction_survives_hamiltonian_cancellation() -> None:
    circuit = Circuit(1).ry(0, theta=0.23)
    observable = _cancellation_hamiltonian()
    p1 = execute_split_real_imag_expectation(circuit, observable, preflight=False)
    p2 = execute_split_real_imag_precision_expectation(
        circuit, observable, preflight=False
    )
    reference = torch.cos(torch.tensor(0.23, dtype=torch.float64))

    p1_error = torch.abs(p1.value.detach().cpu().to(torch.float64) - reference)
    p2_error = torch.abs(p2.cpu_float64() - reference)
    assert p2_error < p1_error / 1000.0
    assert p2.value.high.dtype == torch.float32
    assert p2.value.low.dtype == torch.float32
    assert p2.summary()["reduction_dtype"] == "double_single_fp32"
    assert p2.summary()["convergence_evidence"] is False


def test_p2_parameter_shift_preserves_gradient_residual() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1).ry(0, theta=theta)
    bindings = {"theta": 0.23}
    observable = _cancellation_hamiltonian()
    p1 = parameter_shift_split_real_imag_gradient(
        circuit,
        observable,
        parameter_bindings=bindings,
        preflight=False,
    )
    p2 = parameter_shift_split_real_imag_precision_gradient(
        circuit,
        observable,
        parameter_bindings=bindings,
        preflight=False,
    )
    reference = -torch.sin(torch.tensor(0.23, dtype=torch.float64)).reshape(1)
    p1_error = torch.linalg.vector_norm(
        p1.gradient.detach().cpu().to(torch.float64) - reference
    )
    p2_error = torch.linalg.vector_norm(p2.cpu_float64() - reference)
    assert p2_error < p1_error / 1000.0
    assert p2.parameter_order == ("theta",)
    assert p2.gradient.high.dtype == torch.float32
    assert p2.gradient.low.dtype == torch.float32


def test_p2_rejects_uncertified_accuracy_and_precision_requests() -> None:
    circuit = Circuit(1).h(0)
    observable = pauli_term(1.0, "Z", (0,))
    with pytest.raises(RuntimeError, match="cannot certify"):
        execute_split_real_imag_precision_expectation(
            circuit,
            observable,
            accuracy_requirement=AccuracyRequirementContract(
                max_expectation_abs_error=1e-10
            ),
            preflight=False,
        )
    with pytest.raises(RuntimeError, match="lacks evidence"):
        execute_split_real_imag_precision_expectation(
            circuit,
            observable,
            accuracy_requirement=AccuracyRequirementContract(
                require_convergence_evidence=True
            ),
            preflight=False,
        )
    payload = split_real_imag_p2_precision_plan().to_dict()
    payload["reduction_dtype"] = "float32"
    with pytest.raises(NotImplementedError, match="not executable"):
        execute_split_real_imag_precision_expectation(
            circuit,
            observable,
            precision_plan=PrecisionPlanContract.from_dict(payload),
            preflight=False,
        )


def test_p2_profile_is_forward_only_fp32_and_probes_on_cpu() -> None:
    profile = load_operator_profile("split_real_imag_statevector_p2_precision")
    assert {dtype for item in profile.requirements for dtype in item.dtypes} == {
        "float32"
    }
    assert all(not item.backward for item in profile.requirements)
    report = preflight_split_real_imag_statevector_p2(
        device="cpu", provider="pytorch_cpu_test", refresh=True
    )
    assert report.supported
    assert len(report.evidence_ids) == len(profile.requirements)


def test_p2_sensitive_conformance_improves_on_fp32() -> None:
    report = run_split_real_imag_precision_conformance("cpu", depths=(2,), seeds=(0,))
    report.require_accepted()
    case = report.cases[0]
    assert case.expectation_improvement_factor > 100.0
    assert case.gradient_improvement_factor > 100.0
