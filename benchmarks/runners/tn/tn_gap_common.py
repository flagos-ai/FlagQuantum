"""Framework-neutral contracts for the FlagQuantum/CoTenGra TN gap benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKLOAD_SCHEMA = "flagquantum.tn_gap_workload.v1"
RESULT_SCHEMA = "flagquantum.tn_cotengra_gap.v1"

_DTYPE_BYTES = {
    "complex64": 8,
    "complex128": 16,
    "float32": 4,
    "float64": 8,
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True)
class TNGapWorkload:
    """A tensor network topology with no framework-owned runtime objects."""

    schema_version: str
    name: str
    inputs: tuple[tuple[int, ...], ...]
    shapes: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    dtype: str
    category: str
    identity: str

    @property
    def size_dict(self) -> dict[int, int]:
        dimensions: dict[int, int] = {}
        for labels, shape in zip(self.inputs, self.shapes, strict=True):
            for label, extent in zip(labels, shape, strict=True):
                previous = dimensions.setdefault(int(label), int(extent))
                if previous != int(extent):
                    raise ValueError(
                        f"label {label} has inconsistent extents {previous} and {extent}"
                    )
        return dimensions

    @property
    def element_size(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "inputs": self.inputs,
            "shapes": self.shapes,
            "output": self.output,
            "dtype": self.dtype,
            "category": self.category,
        }

    def validate(self) -> None:
        if self.schema_version != WORKLOAD_SCHEMA:
            raise ValueError(f"unsupported TN gap workload schema {self.schema_version}")
        if not self.name:
            raise ValueError("TN gap workload name must not be empty")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported TN gap workload dtype {self.dtype!r}")
        if not self.inputs or len(self.inputs) != len(self.shapes):
            raise ValueError("TN gap workload inputs and shapes must be non-empty")
        for index, (labels, shape) in enumerate(
            zip(self.inputs, self.shapes, strict=True)
        ):
            if not labels or len(labels) != len(shape):
                raise ValueError(
                    f"TN gap workload input {index} labels/shape do not match"
                )
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"TN gap workload input {index} contains a repeated label"
                )
            if any(int(extent) <= 0 for extent in shape):
                raise ValueError("TN gap workload dimensions must be positive")
        dimensions = self.size_dict
        if len(set(self.output)) != len(self.output):
            raise ValueError("TN gap workload output labels must be unique")
        if any(int(label) not in dimensions for label in self.output):
            raise ValueError("TN gap workload output references an unknown label")
        expected = _sha256(self.identity_payload())
        if self.identity != expected:
            raise ValueError("TN gap workload identity does not match its contents")

    def summary(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.identity_payload(),
            "identity": self.identity,
            "tensor_count": len(self.inputs),
            "label_count": len(self.size_dict),
            "element_size": self.element_size,
        }


def make_workload(
    *,
    name: str,
    inputs: Sequence[Sequence[int]],
    shapes: Sequence[Sequence[int]],
    output: Sequence[int] = (),
    dtype: str = "complex64",
    category: str,
) -> TNGapWorkload:
    payload = {
        "schema_version": WORKLOAD_SCHEMA,
        "name": str(name),
        "inputs": tuple(tuple(int(label) for label in labels) for labels in inputs),
        "shapes": tuple(tuple(int(dim) for dim in shape) for shape in shapes),
        "output": tuple(int(label) for label in output),
        "dtype": str(dtype),
        "category": str(category),
    }
    workload = TNGapWorkload(identity=_sha256(payload), **payload)
    workload.validate()
    return workload


def load_workload(path: str | Path) -> TNGapWorkload:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    workload = TNGapWorkload(
        schema_version=str(payload["schema_version"]),
        name=str(payload["name"]),
        inputs=tuple(tuple(int(label) for label in item) for item in payload["inputs"]),
        shapes=tuple(tuple(int(dim) for dim in item) for item in payload["shapes"]),
        output=tuple(int(label) for label in payload["output"]),
        dtype=str(payload["dtype"]),
        category=str(payload["category"]),
        identity=str(payload["identity"]),
    )
    workload.validate()
    return workload


def write_workload(workload: TNGapWorkload, path: str | Path) -> None:
    workload.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(workload.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_backend_result(
    payload: Mapping[str, Any],
    *,
    expected_backend: str | None = None,
) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unsupported TN gap result schema")
    if expected_backend is not None and payload.get("backend") != expected_backend:
        raise ValueError("TN gap result backend does not match")
    if not payload.get("workload_identity"):
        raise ValueError("TN gap result requires workload identity")
    if payload.get("status") not in {"completed", "failed", "unavailable"}:
        raise ValueError("TN gap result has invalid status")
    if payload.get("comparison_scope") != "planning":
        raise ValueError("stage-A TN gap result must be planning-only")
    if payload.get("distribution_semantics") != "planning_only":
        raise ValueError("TN gap planning result has invalid distribution semantics")
    if payload.get("scalability_claim_allowed") is not False:
        raise ValueError("TN gap planning result cannot allow scalability claims")
    if payload.get("status") == "completed":
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("completed TN gap result requires metrics")
        for field in (
            "search_time_seconds",
            "estimated_flops",
            "largest_intermediate_elements",
        ):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"invalid completed TN gap metric {field}")
        if not isinstance(payload.get("search_budget_compliant"), bool):
            raise ValueError(
                "completed TN gap result requires search_budget_compliant"
            )


def write_result(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_backend_result(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replay_pair_path_metrics(
    workload: TNGapWorkload,
    path: Sequence[Sequence[int]],
) -> dict[str, int]:
    """Score a dynamic pair-position path with one framework-neutral model."""

    workload.validate()
    active = [
        (tuple(labels), tuple(shape))
        for labels, shape in zip(workload.inputs, workload.shapes, strict=True)
    ]
    total_cost = 0
    peak = 0
    total_intermediate = 0
    final_outputs = set(workload.output)
    for pair in path:
        if len(pair) != 2:
            raise ValueError("TN gap pair path entries require two positions")
        left_index, right_index = (int(pair[0]), int(pair[1]))
        if not (
            0 <= left_index < len(active)
            and 0 <= right_index < len(active)
            and left_index != right_index
        ):
            raise ValueError("TN gap pair path position is out of range")
        dimensions: dict[int, int] = {}
        counts: dict[int, int] = {}
        for labels, shape in active:
            for label, extent in zip(labels, shape, strict=True):
                dimensions[label] = extent
                counts[label] = counts.get(label, 0) + 1
        left_labels = active[left_index][0]
        right_labels = active[right_index][0]
        union = tuple(dict.fromkeys(left_labels + right_labels))
        output = tuple(
            label
            for label in union
            if not (
                label not in final_outputs
                and (
                    counts[label] == 1
                    or (
                        label in left_labels
                        and label in right_labels
                        and counts[label] == 2
                    )
                )
            )
        )
        cost = 1
        for label in union:
            cost *= dimensions[label]
        size = 1
        for label in output:
            size *= dimensions[label]
        total_cost += cost
        peak = max(peak, size)
        total_intermediate += size
        node = (output, tuple(dimensions[label] for label in output))
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(node)
    if len(active) != 1:
        raise ValueError("TN gap pair path did not reduce to one value")
    return {
        "estimated_flops": int(total_cost),
        "largest_intermediate_elements": int(peak),
        "total_intermediate_elements": int(total_intermediate),
    }


__all__ = [
    "RESULT_SCHEMA",
    "TNGapWorkload",
    "WORKLOAD_SCHEMA",
    "load_workload",
    "make_workload",
    "replay_pair_path_metrics",
    "validate_backend_result",
    "write_result",
    "write_workload",
]
