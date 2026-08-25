from __future__ import annotations

import math

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import pauli_term
from flagquantum.numerics.double_single import (
    DoubleSingleTensor,
    double_single_sin_cos,
)
from flagquantum.runtime.backends.statevector.split_real_imag_device_double_single import (
    execute_split_real_imag_device_double_single_expectation,
    execute_split_real_imag_device_double_single_statevector,
    parameter_shift_split_real_imag_device_double_single_gradient,
)
from flagquantum.runtime.backends.statevector.split_real_imag_device_double_single_conformance import (
    run_split_real_imag_device_double_single_conformance,
)
from flagquantum.runtime.capabilities import load_operator_profile
from flagquantum.runtime.operator_probes import (
    preflight_split_real_imag_statevector_p4,
)

pytestmark = pytest.mark.unit


def test_device_double_single_trigonometry_breaks_fp32_error_floor() -> None:
    reference = torch.tensor(
        [-1024.0, -math.pi, -0.7, 0.0, 0.7, math.pi, 1024.0],
        dtype=torch.float64,
    )
    sine, cosine = double_single_sin_cos(DoubleSingleTensor.from_float64(reference))
    assert torch.max(torch.abs(sine.to_float64() - torch.sin(reference))) < 1e-11
    assert torch.max(torch.abs(cosine.to_float64() - torch.cos(reference))) < 1e-11


def test_p4_state_uses_device_generated_gates_and_four_fp32_words() -> None:
    circuit = (
        Circuit(2)
        .h(0)
        .ry(1, theta=torch.tensor(0.23, dtype=torch.float32))
        .cx(0, 1)
        .rzz(0, 1, theta=torch.tensor(-0.17, dtype=torch.float32))
    )
    result = execute_split_real_imag_device_double_single_statevector(
        circuit, preflight=False, renormalize_every=2
    )
    reference = (
        Circuit.from_ir(circuit.to_ir(), device="cpu", dtype=torch.complex128)
        .state()
        .reshape(-1)
    )
    assert torch.max(torch.abs(result.cpu_complex128() - reference)) < 1e-11
    assert result.state.real.high.dtype == torch.float32
    assert result.state.real.low.dtype == torch.float32
    assert result.state.imag.high.dtype == torch.float32
    assert result.state.imag.low.dtype == torch.float32
    summary = result.summary()
    assert summary["device_only_double_single_trigonometry"] is True
    assert summary["host_gate_encoding"] is False
    assert summary["parameter_host_fallback"] is False
    assert summary["state_host_fallback"] is False
    assert summary["complex_accelerator_tensor_materialized"] is False


def test_p4_expectation_and_gradient_match_float32_input_reference() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1).ry(0, theta=theta)
    observable = pauli_term(1.0, "Z", (0,))
    angle = torch.tensor(0.23, dtype=torch.float32)
    expectation = execute_split_real_imag_device_double_single_expectation(
        circuit,
        observable,
        parameter_bindings={"theta": angle},
        preflight=False,
    )
    gradient = parameter_shift_split_real_imag_device_double_single_gradient(
        circuit,
        observable,
        parameter_bindings={"theta": angle},
        preflight=False,
    )
    reference = angle.to(torch.float64)
    assert torch.abs(expectation.cpu_float64() - torch.cos(reference)) < 1e-11
    assert torch.max(torch.abs(gradient.cpu_float64() + torch.sin(reference))) < 1e-10
    assert gradient.parameter_order == ("theta",)


def test_p4_device_double_single_binding_avoids_host_ingestion() -> None:
    theta = Parameter("theta")
    pair = DoubleSingleTensor.from_float32(torch.tensor(0.23, dtype=torch.float32))
    result = execute_split_real_imag_device_double_single_expectation(
        Circuit(1).rx(0, theta=theta),
        pauli_term(1.0, "Z", (0,)),
        parameter_bindings={"theta": pair},
        preflight=False,
    )
    assert result.state.summary()["host_scalar_parameter_ingestion"] is False


def test_p4_reports_python_scalar_ingestion_without_calling_it_fallback() -> None:
    result = execute_split_real_imag_device_double_single_statevector(
        Circuit(1).rx(0, theta=0.23), preflight=False
    )
    summary = result.summary()
    assert summary["host_scalar_parameter_ingestion"] is True
    assert summary["parameter_host_fallback"] is False


def test_p4_rejects_hidden_float64_parameter_encoding_and_uncertified_angles() -> None:
    with pytest.raises(TypeError, match="double-precision tensors"):
        execute_split_real_imag_device_double_single_statevector(
            Circuit(1).rx(0, theta=torch.tensor(0.23, dtype=torch.float64)),
            preflight=False,
        )
    with pytest.raises(ValueError, match="certifies"):
        execute_split_real_imag_device_double_single_statevector(
            Circuit(1).rx(0, theta=2048.0), preflight=False
        )


def test_p4_profile_and_small_conformance_pass_on_cpu() -> None:
    profile = load_operator_profile(
        "split_real_imag_statevector_p4_device_double_single"
    )
    assert {dtype for item in profile.requirements for dtype in item.dtypes} == {
        "float32"
    }
    assert all(not item.backward for item in profile.requirements)
    report = preflight_split_real_imag_statevector_p4(
        device="cpu", provider="pytorch_cpu_test", refresh=True
    )
    assert report.supported
    conformance = run_split_real_imag_device_double_single_conformance(
        "cpu",
        depths=(2,),
        seeds=(0,),
        stability_depths=(8,),
        stability_seeds=(0,),
    )
    conformance.require_accepted()
    assert conformance.host_gate_encoding is False
    assert conformance.parameter_host_fallback is False
