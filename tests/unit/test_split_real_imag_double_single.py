from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import pauli_term
from flagquantum.core.numerics import AccuracyRequirementContract, PrecisionPlanContract
from flagquantum.runtime.capabilities import load_operator_profile
from flagquantum.runtime.executors.statevector.split_real_imag_double_single import (
    execute_split_real_imag_double_single_expectation,
    execute_split_real_imag_double_single_statevector,
    parameter_shift_split_real_imag_double_single_gradient,
    split_real_imag_p3_precision_plan,
)
from flagquantum.runtime.executors.statevector.split_real_imag_double_single_conformance import (
    run_split_real_imag_double_single_conformance,
)
from flagquantum.runtime.operator_probes import _preflight_split_real_imag_profile
from flagquantum.simulation.numerics.double_single import DoubleSingleTensor

pytestmark = pytest.mark.unit


def test_double_single_reciprocal_and_sqrt_refine_fp32_seed() -> None:
    reference = torch.tensor([0.3, 1.7, 9.0], dtype=torch.float64)
    value = DoubleSingleTensor.from_float64(reference)
    reciprocal = value.reciprocal().to_float64()
    square_root = value.sqrt().to_float64()
    assert torch.max(torch.abs(reciprocal - torch.reciprocal(reference))) < 1e-13
    assert torch.max(torch.abs(square_root - torch.sqrt(reference))) < 1e-13


def test_p3_state_uses_four_fp32_words_and_matches_complex128() -> None:
    circuit = Circuit(2).h(0).ry(1, theta=0.23).cx(0, 1).rzz(0, 1, theta=-0.17)
    result = execute_split_real_imag_double_single_statevector(
        circuit, preflight=False, renormalize_every=2
    )
    reference = (
        Circuit.from_ir(circuit.to_ir(), device="cpu", dtype=torch.complex128)
        .state()
        .reshape(-1)
    )
    assert torch.max(torch.abs(result.cpu_complex128() - reference)) < 1e-12
    assert result.state.real.high.dtype == torch.float32
    assert result.state.real.low.dtype == torch.float32
    assert result.state.imag.high.dtype == torch.float32
    assert result.state.imag.low.dtype == torch.float32
    assert result.normalization_count == 2
    assert result.summary()["state_word_count_per_amplitude"] == 4
    assert result.summary()["host_gate_encoding"] is True
    assert result.summary()["state_host_fallback"] is False


def test_p3_expectation_and_parameter_shift_break_fp32_error_floor() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1).ry(0, theta=theta)
    observable = pauli_term(1.0, "Z", (0,))
    expectation = execute_split_real_imag_double_single_expectation(
        circuit,
        observable,
        parameter_bindings={"theta": 0.23},
        preflight=False,
    )
    gradient = parameter_shift_split_real_imag_double_single_gradient(
        circuit,
        observable,
        parameter_bindings={"theta": 0.23},
        preflight=False,
    )
    angle = torch.tensor(0.23, dtype=torch.float64)
    assert torch.abs(expectation.cpu_float64() - torch.cos(angle)) < 1e-12
    assert torch.max(torch.abs(gradient.cpu_float64() + torch.sin(angle))) < 1e-12
    assert gradient.parameter_order == ("theta",)


def test_p3_fails_closed_for_unimplemented_claims_and_plans() -> None:
    circuit = Circuit(1).h(0)
    observable = pauli_term(1.0, "Z", (0,))
    with pytest.raises(RuntimeError, match="lacks evidence"):
        execute_split_real_imag_double_single_expectation(
            circuit,
            observable,
            accuracy_requirement=AccuracyRequirementContract(
                require_convergence_evidence=True
            ),
            preflight=False,
        )
    payload = split_real_imag_p3_precision_plan().to_dict()
    payload["gate_generation_dtype"] = "double_single_fp32_device_only"
    with pytest.raises(NotImplementedError, match="not executable"):
        execute_split_real_imag_double_single_expectation(
            circuit,
            observable,
            precision_plan=PrecisionPlanContract.from_dict(payload),
            preflight=False,
        )


def test_p3_profile_is_forward_only_fp32_and_probes_on_cpu() -> None:
    profile = load_operator_profile("split_real_imag_statevector_p3_double_single")
    assert {dtype for item in profile.requirements for dtype in item.dtypes} == {
        "float32"
    }
    assert all(not item.backward for item in profile.requirements)
    report = _preflight_split_real_imag_profile(
        profile.name, device="cpu", provider="pytorch_cpu_test", refresh=True
    )
    assert report.supported
    assert len(report.evidence_ids) == len(profile.requirements)


def test_p3_depth_conformance_improves_over_p2() -> None:
    report = run_split_real_imag_double_single_conformance(
        "cpu", depths=(2,), seeds=(0,)
    )
    report.require_accepted()
    case = report.cases[0]
    assert case.max_state_abs_error < 1e-10
    assert case.expectation_improvement_factor > 100.0
    assert case.gradient_improvement_factor > 100.0
    assert report.host_gate_encoding is True
    assert report.state_host_fallback is False
    assert report.convergence_certification is False
