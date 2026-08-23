"""Versioned numerical requirements and precision plans.

The requirement describes *what quality is acceptable*.  The plan describes
*how a platform intends to satisfy it*.  Keeping them separate lets vendor
backends change representations without changing algorithm semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Mapping

NUMERICAL_CONTRACT_VERSION = "1.0"


class AccuracyMode(str, Enum):
    STRICT = "strict"
    ADAPTIVE = "adaptive"
    FAST = "fast"


class ComplexRepresentation(str, Enum):
    NATIVE_COMPLEX = "native_complex"
    SPLIT_REAL_IMAG = "split_real_imag"
    DOUBLE_SINGLE_FP32 = "double_single_fp32"


class RefinementStrategy(str, Enum):
    NONE = "none"
    COMPENSATED_REDUCTION = "compensated_reduction"
    ADAPTIVE_BLOCK_REPLAY = "adaptive_block_replay"


class _NumericalContract:
    contract_version: str
    KIND: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.KIND
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def _validated_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        kind = data.pop("kind", cls.KIND)
        if kind != cls.KIND:
            raise ValueError(
                f"expected numerical contract kind {cls.KIND!r}, got {kind!r}"
            )
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown {cls.KIND} fields: {unknown}")
        return data


def _validate_optional_nonnegative(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class AccuracyRequirementContract(_NumericalContract):
    """Numerical acceptance criteria specified before plan selection.

    This is distinct from ``core.contracts.AccuracyContract``, which records
    an observed result after execution.
    """

    mode: str = AccuracyMode.STRICT.value
    max_norm_drift: float | None = None
    max_expectation_abs_error: float | None = None
    max_expectation_rel_error: float | None = None
    max_gradient_rel_error: float | None = None
    min_gradient_cosine_similarity: float | None = None
    max_state_infidelity: float | None = None
    max_decomposition_residual: float | None = None
    max_truncation_error: float | None = None
    require_determinism: bool = False
    require_convergence_evidence: bool = False
    contract_version: str = NUMERICAL_CONTRACT_VERSION

    KIND: ClassVar[str] = "accuracy_requirement"

    def __post_init__(self) -> None:
        if self.mode not in {item.value for item in AccuracyMode}:
            raise ValueError(f"unsupported accuracy mode: {self.mode!r}")
        if self.contract_version != NUMERICAL_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported numerical contract version: {self.contract_version!r}"
            )
        for name in (
            "max_norm_drift",
            "max_expectation_abs_error",
            "max_expectation_rel_error",
            "max_gradient_rel_error",
            "max_state_infidelity",
            "max_decomposition_residual",
            "max_truncation_error",
        ):
            _validate_optional_nonnegative(name, getattr(self, name))
        cosine = self.min_gradient_cosine_similarity
        if cosine is not None and (
            not math.isfinite(cosine) or cosine < -1.0 or cosine > 1.0
        ):
            raise ValueError("min_gradient_cosine_similarity must be in [-1, 1]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AccuracyRequirementContract":
        return cls(**cls._validated_payload(payload))


@dataclass(frozen=True)
class PrecisionPlanContract(_NumericalContract):
    """Auditable dtype and representation choices for one execution plan."""

    complex_representation: str = ComplexRepresentation.NATIVE_COMPLEX.value
    parameter_dtype: str = "float64"
    gate_generation_dtype: str = "complex128"
    state_storage_dtype: str = "complex128"
    kernel_compute_dtype: str = "complex128"
    reduction_dtype: str = "complex128"
    decomposition_dtype: str = "complex128"
    gradient_dtype: str = "float64"
    optimizer_master_dtype: str = "float64"
    communication_dtype: str = "complex128"
    checkpoint_dtype: str = "complex128"
    refinement: str = RefinementStrategy.NONE.value
    allow_dtype_demotion: bool = False
    contract_version: str = NUMERICAL_CONTRACT_VERSION

    KIND: ClassVar[str] = "precision_plan"

    def __post_init__(self) -> None:
        if self.complex_representation not in {
            item.value for item in ComplexRepresentation
        }:
            raise ValueError(
                f"unsupported complex representation: {self.complex_representation!r}"
            )
        if self.refinement not in {item.value for item in RefinementStrategy}:
            raise ValueError(f"unsupported refinement strategy: {self.refinement!r}")
        if self.contract_version != NUMERICAL_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported numerical contract version: {self.contract_version!r}"
            )
        for name in (
            "parameter_dtype",
            "gate_generation_dtype",
            "state_storage_dtype",
            "kernel_compute_dtype",
            "reduction_dtype",
            "decomposition_dtype",
            "gradient_dtype",
            "optimizer_master_dtype",
            "communication_dtype",
            "checkpoint_dtype",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty dtype name")
        if (
            self.complex_representation
            == ComplexRepresentation.DOUBLE_SINGLE_FP32.value
        ):
            incompatible = {
                name: getattr(self, name)
                for name in ("state_storage_dtype", "kernel_compute_dtype")
                if getattr(self, name) != "float32"
            }
            if incompatible:
                raise ValueError(
                    "double_single_fp32 requires float32 storage and compute; "
                    f"got {incompatible}"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrecisionPlanContract":
        return cls(**cls._validated_payload(payload))


def default_accuracy_requirement(dtype: str) -> AccuracyRequirementContract:
    """Return the shipped acceptance gate for a native statevector dtype."""

    normalized = str(dtype).removeprefix("torch.")
    if normalized == "complex128":
        return AccuracyRequirementContract(
            mode=AccuracyMode.STRICT.value,
            max_norm_drift=1e-11,
            max_expectation_abs_error=1e-11,
            max_gradient_rel_error=1e-9,
            min_gradient_cosine_similarity=0.999999999,
            max_state_infidelity=1e-11,
            require_determinism=True,
            require_convergence_evidence=True,
        )
    if normalized == "complex64":
        return AccuracyRequirementContract(
            mode=AccuracyMode.ADAPTIVE.value,
            max_norm_drift=5e-5,
            max_expectation_abs_error=5e-5,
            max_gradient_rel_error=2e-4,
            min_gradient_cosine_similarity=0.999,
            max_state_infidelity=5e-6,
            require_determinism=True,
            require_convergence_evidence=True,
        )
    raise ValueError(f"unsupported statevector dtype: {dtype!r}")


def default_precision_plan(dtype: str) -> PrecisionPlanContract:
    """Describe the native-complex implementation used by local statevector."""

    normalized = str(dtype).removeprefix("torch.")
    real_dtype = {"complex64": "float32", "complex128": "float64"}.get(normalized)
    if real_dtype is None:
        raise ValueError(f"unsupported statevector dtype: {dtype!r}")
    return PrecisionPlanContract(
        parameter_dtype=real_dtype,
        gate_generation_dtype=normalized,
        state_storage_dtype=normalized,
        kernel_compute_dtype=normalized,
        reduction_dtype=normalized,
        decomposition_dtype=normalized,
        gradient_dtype=real_dtype,
        optimizer_master_dtype=real_dtype,
        communication_dtype=normalized,
        checkpoint_dtype=normalized,
    )


def coerce_accuracy_requirement(
    value: AccuracyRequirementContract | Mapping[str, Any] | None,
    *,
    dtype: str,
) -> AccuracyRequirementContract:
    if value is None:
        return default_accuracy_requirement(dtype)
    if isinstance(value, AccuracyRequirementContract):
        return value
    return AccuracyRequirementContract.from_dict(value)


def coerce_precision_plan(
    value: PrecisionPlanContract | Mapping[str, Any] | None,
    *,
    dtype: str,
) -> PrecisionPlanContract:
    plan = (
        default_precision_plan(dtype)
        if value is None
        else (
            value
            if isinstance(value, PrecisionPlanContract)
            else PrecisionPlanContract.from_dict(value)
        )
    )
    expected = default_precision_plan(dtype)
    implemented = (
        "complex_representation",
        "parameter_dtype",
        "gate_generation_dtype",
        "state_storage_dtype",
        "kernel_compute_dtype",
        "reduction_dtype",
        "decomposition_dtype",
        "gradient_dtype",
        "optimizer_master_dtype",
        "communication_dtype",
        "checkpoint_dtype",
        "refinement",
        "allow_dtype_demotion",
    )
    differences = {
        name: (getattr(plan, name), getattr(expected, name))
        for name in implemented
        if getattr(plan, name) != getattr(expected, name)
    }
    if differences:
        raise NotImplementedError(
            "local statevector cannot honestly execute the requested precision "
            f"plan yet; requested/implemented differences: {differences}"
        )
    return plan


__all__ = [
    "NUMERICAL_CONTRACT_VERSION",
    "AccuracyMode",
    "AccuracyRequirementContract",
    "ComplexRepresentation",
    "PrecisionPlanContract",
    "RefinementStrategy",
    "coerce_accuracy_requirement",
    "coerce_precision_plan",
    "default_accuracy_requirement",
    "default_precision_plan",
]
