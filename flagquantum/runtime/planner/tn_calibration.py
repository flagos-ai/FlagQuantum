"""Versioned Runtime tensor-network working-set calibration records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

TN_WORKING_SET_CALIBRATION_VERSION = "flagquantum.tn_working_set_calibration.v1"


@dataclass(frozen=True)
class TNWorkingSetCalibration:
    """Conservative measured-to-predicted memory ratio for one TN scope."""

    version: str
    identity: str
    measurement_digest: str
    accelerator_name: str
    complex_bytes: int
    world_size: int
    topology_class: str
    sample_count: int
    distinct_prediction_count: int
    maximum_allocated_to_predicted_ratio: float
    maximum_reserved_to_predicted_ratio: float
    fixed_reserved_overhead_bytes: int
    recommended_safety_factor: float
    evidence_level: str
    passed: bool
    blockers: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return asdict(self)

    def applies_to(
        self,
        *,
        accelerator_name: str | None,
        complex_bytes: int,
        world_size: int,
    ) -> bool:
        return (
            self.passed
            and accelerator_name is not None
            and self.accelerator_name == accelerator_name.strip()
            and self.complex_bytes == int(complex_bytes)
            and self.world_size == int(world_size)
        )

    def apply(self, predicted_bytes: int) -> int:
        if predicted_bytes < 0:
            raise ValueError("TN predicted bytes must be non-negative")
        return self.fixed_reserved_overhead_bytes + ceil(
            predicted_bytes * self.recommended_safety_factor
        )


def build_tn_working_set_calibration(
    measurements: Sequence[Mapping[str, Any]],
    *,
    accelerator_name: str,
    complex_bytes: int,
    world_size: int,
    topology_class: str,
    minimum_samples: int = 3,
    minimum_distinct_predictions: int = 3,
    safety_margin: float = 1.1,
    evidence_level: str = "development_hardware_calibration",
) -> TNWorkingSetCalibration:
    """Build a deterministic fail-closed calibration from CUDA peak records."""

    if not accelerator_name.strip():
        raise ValueError("TN calibration accelerator name must be non-empty")
    if complex_bytes not in {8, 16}:
        raise ValueError("TN calibration complex bytes must be 8 or 16")
    if world_size < 1:
        raise ValueError("TN calibration world size must be positive")
    if not topology_class.strip():
        raise ValueError("TN calibration topology class must be non-empty")
    if minimum_samples < 1:
        raise ValueError("TN calibration minimum samples must be positive")
    if minimum_distinct_predictions < 1:
        raise ValueError("TN calibration minimum distinct predictions must be positive")
    if safety_margin < 1.0:
        raise ValueError("TN calibration safety margin must be at least one")

    normalized = tuple(
        sorted(
            (
                {
                    "predicted_working_set_bytes": int(
                        item["predicted_working_set_bytes"]
                    ),
                    "cuda_peak_allocated_bytes": int(item["cuda_peak_allocated_bytes"]),
                    "cuda_peak_reserved_bytes": int(item["cuda_peak_reserved_bytes"]),
                }
                for item in measurements
            ),
            key=lambda item: (
                item["predicted_working_set_bytes"],
                item["cuda_peak_allocated_bytes"],
                item["cuda_peak_reserved_bytes"],
            ),
        )
    )
    for item in normalized:
        if item["predicted_working_set_bytes"] <= 0:
            raise ValueError("TN calibration predictions must be positive")
        if item["cuda_peak_allocated_bytes"] <= 0:
            raise ValueError("TN calibration allocated peaks must be positive")
        if item["cuda_peak_reserved_bytes"] < item["cuda_peak_allocated_bytes"]:
            raise ValueError(
                "TN calibration reserved peaks cannot be below allocated peaks"
            )

    allocated_ratios = tuple(
        item["cuda_peak_allocated_bytes"] / item["predicted_working_set_bytes"]
        for item in normalized
    )
    reserved_ratios = tuple(
        item["cuda_peak_reserved_bytes"] / item["predicted_working_set_bytes"]
        for item in normalized
    )
    max_allocated = max(allocated_ratios, default=0.0)
    max_reserved = max(reserved_ratios, default=0.0)
    fixed_reserved_overhead = min(
        (item["cuda_peak_reserved_bytes"] for item in normalized),
        default=0,
    )
    residual_reserved_ratios = tuple(
        max(
            0,
            item["cuda_peak_reserved_bytes"] - fixed_reserved_overhead,
        )
        / item["predicted_working_set_bytes"]
        for item in normalized
    )
    distinct_predictions = len(
        {item["predicted_working_set_bytes"] for item in normalized}
    )
    blockers = []
    if len(normalized) < minimum_samples:
        blockers.append("insufficient_hardware_calibration_samples")
    if distinct_predictions < minimum_distinct_predictions:
        blockers.append("insufficient_working_set_shape_diversity")
    recommended_safety_factor = max(
        1.0,
        max(residual_reserved_ratios, default=0.0) * safety_margin,
    )
    calibration_blockers = tuple(blockers)
    measurement_digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "version": TN_WORKING_SET_CALIBRATION_VERSION,
        "accelerator_name": accelerator_name.strip(),
        "complex_bytes": int(complex_bytes),
        "world_size": int(world_size),
        "topology_class": topology_class.strip(),
        "sample_count": len(normalized),
        "distinct_prediction_count": distinct_predictions,
        "maximum_allocated_to_predicted_ratio": max_allocated,
        "maximum_reserved_to_predicted_ratio": max_reserved,
        "fixed_reserved_overhead_bytes": fixed_reserved_overhead,
        "recommended_safety_factor": recommended_safety_factor,
        "evidence_level": evidence_level,
        "passed": not blockers,
        "blockers": calibration_blockers,
        "measurement_digest": measurement_digest,
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TNWorkingSetCalibration(
        version=TN_WORKING_SET_CALIBRATION_VERSION,
        identity=identity,
        measurement_digest=measurement_digest,
        accelerator_name=accelerator_name.strip(),
        complex_bytes=int(complex_bytes),
        world_size=int(world_size),
        topology_class=topology_class.strip(),
        sample_count=len(normalized),
        distinct_prediction_count=distinct_predictions,
        maximum_allocated_to_predicted_ratio=max_allocated,
        maximum_reserved_to_predicted_ratio=max_reserved,
        fixed_reserved_overhead_bytes=fixed_reserved_overhead,
        recommended_safety_factor=recommended_safety_factor,
        evidence_level=evidence_level,
        passed=not blockers,
        blockers=calibration_blockers,
    )


def load_tn_working_set_calibration(
    path: str | Path,
) -> TNWorkingSetCalibration:
    """Load a JSON calibration without accepting unknown execution payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = TNWorkingSetCalibration.__dataclass_fields__
    try:
        selected: dict[str, Any] = {name: payload[name] for name in fields}
        selected["blockers"] = tuple(selected["blockers"])
        calibration = TNWorkingSetCalibration(
            version=cast(str, selected["version"]),
            identity=cast(str, selected["identity"]),
            measurement_digest=cast(str, selected["measurement_digest"]),
            accelerator_name=cast(str, selected["accelerator_name"]),
            complex_bytes=cast(int, selected["complex_bytes"]),
            world_size=cast(int, selected["world_size"]),
            topology_class=cast(str, selected["topology_class"]),
            sample_count=cast(int, selected["sample_count"]),
            distinct_prediction_count=cast(int, selected["distinct_prediction_count"]),
            maximum_allocated_to_predicted_ratio=cast(
                float, selected["maximum_allocated_to_predicted_ratio"]
            ),
            maximum_reserved_to_predicted_ratio=cast(
                float, selected["maximum_reserved_to_predicted_ratio"]
            ),
            fixed_reserved_overhead_bytes=cast(
                int, selected["fixed_reserved_overhead_bytes"]
            ),
            recommended_safety_factor=cast(
                float, selected["recommended_safety_factor"]
            ),
            evidence_level=cast(str, selected["evidence_level"]),
            passed=cast(bool, selected["passed"]),
            blockers=cast(tuple[str, ...], selected["blockers"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid TN working-set calibration record") from error
    expected_identity = calibration.identity
    identity_payload = calibration.summary()
    identity_payload.pop("identity")
    computed_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if computed_identity != expected_identity:
        raise ValueError("TN working-set calibration identity mismatch")
    return calibration


__all__ = (
    "TNWorkingSetCalibration",
    "TN_WORKING_SET_CALIBRATION_VERSION",
    "build_tn_working_set_calibration",
    "load_tn_working_set_calibration",
)
