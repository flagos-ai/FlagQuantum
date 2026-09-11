"""Persistent contraction-plan cache for tensor-network execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ....simulation.tensor_network.models import (
    PairContractionStep,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from ....version import __version__

_PERSISTENT_PLAN_SCHEMA = "flagquantum.distributed_tn_plan.v1"


@contextmanager
def _plan_file_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize cache writers while allowing concurrent readers."""

    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _persistent_plan_key(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None,
    max_intermediate_bytes: int | None,
    sliced_labels: Sequence[int] | None,
) -> str:
    payload = {
        "schema": _PERSISTENT_PLAN_SCHEMA,
        "nodes": [
            {
                "name": node.name,
                "labels": list(node.labels),
                "shape": list(node.tensor.shape),
                "element_size": node.tensor.element_size(),
            }
            for node in nodes
        ],
        "output_labels": list(output_labels),
        "max_intermediate_size": max_intermediate_size,
        "max_intermediate_bytes": max_intermediate_bytes,
        "sliced_labels": None if sliced_labels is None else list(sliced_labels),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pair_contraction_step_from_payload(
    payload: Mapping[str, Any],
) -> PairContractionStep:
    """Restore tuple-valued fields erased by JSON serialization."""

    return PairContractionStep(
        **{
            **payload,
            "left_labels": tuple(payload["left_labels"]),
            "right_labels": tuple(payload["right_labels"]),
            "output_labels": tuple(payload["output_labels"]),
            "output_shape": tuple(payload["output_shape"]),
        }
    )


def _load_persistent_plan(
    path: str | Path,
    *,
    expected_key: str,
) -> tuple[TensorNetworkSlicingPlan, tuple[PairContractionStep, ...]] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with _plan_file_lock(target, exclusive=False):
            payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _PERSISTENT_PLAN_SCHEMA
        or payload.get("cache_key") != expected_key
        or payload.get("flagquantum_version") != __version__
        or payload.get("torch_version") != torch.__version__
    ):
        return None
    try:
        slicing_payload = dict(payload["slicing"])
        for field in ("sliced_labels", "slice_shape"):
            slicing_payload[field] = tuple(slicing_payload[field])
        slicing_payload["contraction_path"] = tuple(
            _pair_contraction_step_from_payload(item)
            for item in slicing_payload["contraction_path"]
        )
        slicing = TensorNetworkSlicingPlan(**slicing_payload)
        steps = tuple(
            _pair_contraction_step_from_payload(item) for item in payload["steps"]
        )
    except (TypeError, ValueError, KeyError):
        return None
    return slicing, steps


def _write_persistent_plan(
    path: str | Path,
    *,
    cache_key: str,
    slicing: TensorNetworkSlicingPlan,
    steps: Sequence[PairContractionStep],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _PERSISTENT_PLAN_SCHEMA,
        "flagquantum_version": __version__,
        "torch_version": torch.__version__,
        "cache_key": cache_key,
        "slicing": asdict(slicing),
        "steps": [asdict(step) for step in steps],
    }
    with _plan_file_lock(target, exclusive=True):
        temporary = target.with_suffix(
            target.suffix + f".{hashlib.sha256(cache_key.encode()).hexdigest()[:8]}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
