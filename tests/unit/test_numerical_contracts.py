from __future__ import annotations

import pytest

from flagquantum.core.numerics import (
    AccuracyRequirementContract,
    PrecisionPlanContract,
)
from flagquantum.runtime.fallback import FallbackPolicy


def test_accuracy_requirement_round_trip_and_hash_are_stable() -> None:
    contract = AccuracyRequirementContract(
        mode="adaptive",
        max_norm_drift=1e-8,
        max_expectation_abs_error=1e-7,
        require_convergence_evidence=True,
    )
    restored = AccuracyRequirementContract.from_dict(contract.to_dict())
    assert restored == contract
    assert restored.content_hash() == contract.content_hash()


def test_accuracy_requirement_rejects_invalid_tolerances() -> None:
    with pytest.raises(ValueError, match="max_norm_drift"):
        AccuracyRequirementContract(max_norm_drift=-1.0)
    with pytest.raises(ValueError, match="cosine"):
        AccuracyRequirementContract(min_gradient_cosine_similarity=1.1)


def test_precision_plan_is_versioned_and_strictly_serialized() -> None:
    plan = PrecisionPlanContract(
        complex_representation="split_real_imag",
        state_storage_dtype="float32",
        kernel_compute_dtype="float32",
        reduction_dtype="float64",
        allow_dtype_demotion=True,
    )
    payload = plan.to_dict()
    assert payload["kind"] == "precision_plan"
    assert PrecisionPlanContract.from_dict(payload) == plan
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown precision_plan fields"):
        PrecisionPlanContract.from_dict(payload)


def test_double_single_requires_float32_storage_and_compute() -> None:
    with pytest.raises(ValueError, match="double_single_fp32"):
        PrecisionPlanContract(
            complex_representation="double_single_fp32",
            state_storage_dtype="complex64",
            kernel_compute_dtype="float32",
        )


def test_fallback_policy_is_never_an_implicit_boolean() -> None:
    assert FallbackPolicy.normalize("forbid") is FallbackPolicy.FORBID
    with pytest.raises(ValueError, match="unsupported fallback policy"):
        FallbackPolicy.normalize("silent_cpu")
