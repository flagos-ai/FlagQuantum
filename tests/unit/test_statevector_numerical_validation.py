from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.numerics import (
    PrecisionPlanContract,
    coerce_precision_plan,
    default_accuracy_requirement,
    default_precision_plan,
)
from flagquantum.runtime.numerical_validation import certify_statevector_local_p0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_default_statevector_contracts_match_native_dtype(dtype: str) -> None:
    accuracy = default_accuracy_requirement(dtype)
    precision = default_precision_plan(dtype)

    assert precision.state_storage_dtype == dtype
    assert precision.kernel_compute_dtype == dtype
    assert accuracy.require_convergence_evidence
    assert accuracy.require_determinism


def test_unimplemented_precision_plan_is_rejected() -> None:
    plan = PrecisionPlanContract(
        state_storage_dtype="complex64",
        kernel_compute_dtype="complex64",
    )

    with pytest.raises(NotImplementedError, match="cannot honestly execute"):
        coerce_precision_plan(plan, dtype="complex64")


def test_complex128_fixed_gate_is_not_promoted_from_complex64() -> None:
    circuit = Circuit(1, dtype=torch.complex128)
    state = circuit.h(0).state()

    assert state.dtype == torch.complex128
    assert abs(float(torch.linalg.vector_norm(state).item()) - 1.0) < 1e-14


def test_complex128_python_float_parameter_is_materialized_in_float64() -> None:
    theta = 0.371
    state = Circuit(1, dtype=torch.complex128).ry(0, theta=theta).state()[0]
    expected = torch.tensor(
        [
            torch.cos(torch.tensor(theta / 2, dtype=torch.float64)),
            torch.sin(torch.tensor(theta / 2, dtype=torch.float64)),
        ],
        dtype=torch.complex128,
    )

    assert torch.equal(state, expected)


def test_cswap_exchanges_targets_only_when_control_is_one() -> None:
    state = Circuit(3, dtype=torch.complex128).x(0).x(2).cswap(0, 1, 2).state()[0]
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[6] = 1.0

    assert torch.equal(state, expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cpu_statevector_certification_passes(dtype: torch.dtype) -> None:
    dtype_name = str(dtype).removeprefix("torch.")
    report = certify_statevector_local_p0(
        device="cpu",
        dtype=dtype,
        provider="pytorch_cpu_test",
        accuracy_requirement=default_accuracy_requirement(dtype_name),
        precision_plan=default_precision_plan(dtype_name),
        refresh=True,
    )

    assert report.passed
    assert report.reference == "cpu_complex128"
    assert report.deterministic
    assert report.convergence_evidence
    assert report.metrics["gradient_cosine_similarity"] > 0.999
