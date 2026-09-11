from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit
from flagquantum.runtime.capabilities import load_operator_profile
from flagquantum.runtime.executors.statevector.split_real_imag import (
    SPLIT_REAL_IMAG_SUPPORTED_GATES,
    execute_split_real_imag_statevector,
)
from flagquantum.runtime.operator_probes import (
    _preflight_split_real_imag_profile,
)
from flagquantum.simulation.statevector.split_real_imag import (
    double_single_pauli_term_expectation,
    pauli_term_expectation,
    run_split_real_imag_statevector,
)

pytestmark = pytest.mark.unit


_PARAMS = {
    "rx": {"theta": 0.37},
    "ry": {"theta": -0.23},
    "rz": {"theta": 0.19},
    "phase": {"theta": -0.41},
    "u1": {"theta": 0.29},
    "u2": {"phi": 0.13, "lbd": -0.31},
    "u3": {"theta": 0.47, "phi": -0.17, "lbd": 0.22},
    "crx": {"theta": 0.37},
    "cry": {"theta": -0.23},
    "crz": {"theta": 0.19},
    "cphase": {"theta": -0.41},
    "rxx": {"theta": 0.37},
    "ryy": {"theta": -0.23},
    "rzz": {"theta": 0.19},
}
_TWO_QUBIT = {
    "cx",
    "cy",
    "cz",
    "swap",
    "crx",
    "cry",
    "crz",
    "cphase",
    "rxx",
    "ryy",
    "rzz",
}


def test_simulation_kernel_runs_a_zero_state_circuit_directly() -> None:
    circuit = Circuit(2).h(0).cx(0, 1)

    real, imag = run_split_real_imag_statevector(
        circuit.to_ir(), device=torch.device("cpu")
    )

    expected = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    torch.testing.assert_close(real, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(imag, torch.zeros(4), atol=0.0, rtol=0.0)


def test_simulation_pauli_term_kernel_uses_split_tensors_directly() -> None:
    real, imag = run_split_real_imag_statevector(
        Circuit(1).h(0).to_ir(), device=torch.device("cpu")
    )

    value = pauli_term_expectation(
        real,
        imag,
        ((0, "x"),),
        0.5,
        n_wires=1,
    )

    torch.testing.assert_close(value, torch.tensor(0.5), atol=2e-6, rtol=2e-6)


def test_simulation_double_single_pauli_reduction_uses_split_tensors() -> None:
    real, imag = run_split_real_imag_statevector(
        Circuit(1).h(0).to_ir(), device=torch.device("cpu")
    )

    value = double_single_pauli_term_expectation(
        real,
        imag,
        ((0, "x"),),
        0.5,
        n_wires=1,
    )

    torch.testing.assert_close(
        value.to_float64(), torch.tensor(0.5, dtype=torch.float64)
    )


@pytest.mark.parametrize("name", sorted(SPLIT_REAL_IMAG_SUPPORTED_GATES))
def test_supported_gate_matches_complex128_reference(name: str) -> None:
    circuit = Circuit(2, device="cpu", dtype=torch.complex128)
    circuit.h(0)
    circuit.ry(1, theta=0.27)
    wires = (0, 1) if name in _TWO_QUBIT else 1
    circuit.gate(name, wires, **_PARAMS.get(name, {}))

    result = execute_split_real_imag_statevector(
        circuit.to_ir(), device="cpu", preflight=False
    )

    torch.testing.assert_close(
        result.cpu_complex128(), circuit.state().reshape(-1), atol=2e-6, rtol=2e-6
    )
    assert result.real.dtype == torch.float32
    assert result.imag.dtype == torch.float32
    assert result.real.device.type == "cpu"


def test_executor_does_not_materialize_complex_tensor(monkeypatch) -> None:
    circuit = Circuit(2).h(0).cx(0, 1).rz(1, theta=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("complex tensor creation entered split executor")

    monkeypatch.setattr(torch, "complex", forbidden)
    result = execute_split_real_imag_statevector(
        circuit.to_ir(), device="cpu", preflight=False
    )

    assert result.probabilities().sum().item() == pytest.approx(1.0, abs=2e-6)
    assert result.summary()["complex_accelerator_tensor_materialized"] is False
    assert result.summary()["flagquantum_host_fallback"] is False
    assert result.summary()["provider_internal_route_audited"] is False


def test_executor_rejects_gradients_and_custom_matrices() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    with pytest.raises(NotImplementedError, match="forward-only"):
        execute_split_real_imag_statevector(
            Circuit(1).rx(0, theta=theta).to_ir(), preflight=False
        )

    custom = Circuit(1).any(
        0, unitary=torch.eye(2, dtype=torch.complex64), name="custom"
    )
    with pytest.raises(NotImplementedError, match="custom matrices"):
        execute_split_real_imag_statevector(custom.to_ir(), preflight=False)


def test_executor_rejects_nonzero_input_and_batches() -> None:
    inputs = torch.tensor([[0.0, 1.0]], dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match="input only"):
        execute_split_real_imag_statevector(Circuit(1, inputs=inputs), preflight=False)
    with pytest.raises(NotImplementedError, match="batch size one"):
        execute_split_real_imag_statevector(Circuit(1, bsz=2), preflight=False)


def test_split_profile_is_float32_only_and_cpu_probe_passes() -> None:
    profile = load_operator_profile("split_real_imag_statevector_p0")
    assert profile.representation == "split_real_imag_statevector"
    assert {dtype for item in profile.requirements for dtype in item.dtypes} == {
        "float32"
    }
    assert all(not item.backward for item in profile.requirements)

    report = _preflight_split_real_imag_profile(
        profile.name, device="cpu", provider="pytorch_cpu_test", refresh=True
    )
    assert report.supported
    assert report.required_dtypes == ("float32",)
    assert len(report.evidence_ids) == len(profile.requirements)
