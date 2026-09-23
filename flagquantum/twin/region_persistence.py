"""Provider-neutral persistence for composed regional Twin models.

A composed :class:`~flagquantum.twin.TwinRegionModel` is normally rebuilt from
its source cells, each of which needs its own Twin and support artifact. This
module persists the composed model itself: the structural region and the frozen
composed Twin are written through their existing authoritative serializers, so
one released regional model can be loaded in another process and passed
directly to ``TwinRegionRelease.assess``.

The artifact is a model artifact. It carries no hardware task receipt,
credential, raw provider response, validation evidence, confidence or error
claim, release decision, routing authorization, or application state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ..noise import NoiseModel
from ._atomic import write_once
from .candidate import _matches_canonical_payload
from .model import QPUDigitalTwin
from .persistence import _require_fields, _snapshot_from_dict
from .region import TwinConnectedRegion
from .region_model import TwinRegionModel, _region_profile_source

_ARTIFACT_SCHEMA = "flagquantum.twin_region_model_artifact.v1"
_ARTIFACT_FIELDS = {"schema", "region", "twin"}
_REGION_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "captured_at",
    "physical_qubits",
    "directed_couplers",
    "supported_operations",
    "maximum_instruction_count",
    "maximum_circuit_depth",
    "source_snapshot_identities",
    "source_support_identities",
}
_FROZEN_TWIN_FIELDS = {"snapshot", "noise_model"}

_INVALID_MEMBERS = (TypeError, ValueError, KeyError, AttributeError, IndexError)


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _region_from_dict(payload: Mapping[str, Any]) -> TwinConnectedRegion:
    _require_fields(payload, _REGION_FIELDS, name="Twin connected region")
    try:
        region = TwinConnectedRegion(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            captured_at=str(payload["captured_at"]),
            physical_qubits=tuple(int(qubit) for qubit in payload["physical_qubits"]),
            directed_couplers=tuple(
                (int(edge[0]), int(edge[1])) for edge in payload["directed_couplers"]
            ),
            supported_operations=tuple(
                str(operation) for operation in payload["supported_operations"]
            ),
            maximum_instruction_count=int(payload["maximum_instruction_count"]),
            maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
            source_snapshot_identities=tuple(
                str(identity) for identity in payload["source_snapshot_identities"]
            ),
            source_support_identities=tuple(
                str(identity) for identity in payload["source_support_identities"]
            ),
        )
    except _INVALID_MEMBERS as error:
        raise ValueError("Invalid Twin connected region") from error
    if not _matches_canonical_payload(region.to_dict(), payload):
        raise ValueError("Twin connected region is not in canonical v1 form")
    return region


def _frozen_twin_from_dict(payload: Mapping[str, Any]) -> QPUDigitalTwin:
    _require_fields(payload, _FROZEN_TWIN_FIELDS, name="Twin region-model twin")
    snapshot_payload = payload["snapshot"]
    noise_payload = payload["noise_model"]
    if not isinstance(snapshot_payload, Mapping) or not isinstance(
        noise_payload, Mapping
    ):
        raise ValueError("Twin region-model twin members must be JSON objects")
    try:
        noise_model = NoiseModel.from_dict(noise_payload)
    except _INVALID_MEMBERS as error:
        raise ValueError("Invalid Twin region-model noise model") from error
    if not _matches_canonical_payload(noise_model.to_dict(), noise_payload):
        raise ValueError("Twin region-model noise model is not in canonical v1 form")
    try:
        return QPUDigitalTwin(
            snapshot=_snapshot_from_dict(snapshot_payload),
            noise_model=noise_model,
        )
    except _INVALID_MEMBERS as error:
        raise ValueError("Invalid Twin region-model twin") from error


def _to_dict(region_twin: TwinRegionModel) -> dict[str, Any]:
    if not isinstance(region_twin, TwinRegionModel):
        raise TypeError("region_twin must be a TwinRegionModel")
    region = region_twin.region
    twin = region_twin.twin
    if not isinstance(region, TwinConnectedRegion) or not isinstance(
        twin, QPUDigitalTwin
    ):
        raise ValueError("Twin region model does not hold a region and a frozen Twin")
    if twin.noise_model.identity != twin.snapshot.noise_model_identity:
        raise ValueError("regional Twin noise model changed after it was frozen")
    profile = twin.noise_model.device_profile
    if profile is None or profile.identity != twin.snapshot.calibration_identity:
        raise ValueError("regional Twin calibration changed after it was frozen")
    # The composed calibration is bound to the region it was composed for, so a
    # reordered mapping, a retargeted region, or a rewritten source identity
    # cannot survive this check even when the tampered payload stays canonical.
    if profile.source != _region_profile_source(region.identity) or (
        profile.captured_at != region.captured_at
    ):
        raise ValueError("regional Twin calibration does not match its region")
    if {calibration.wire for calibration in profile.qubits} != set(
        range(len(region.physical_qubits))
    ):
        raise ValueError("regional Twin calibration does not cover every region wire")
    return {
        "schema": _ARTIFACT_SCHEMA,
        "region": region.to_dict(),
        "twin": {
            "snapshot": twin.snapshot.to_dict(),
            "noise_model": twin.noise_model.to_dict(),
        },
    }


def load_region_twin(path: str | PathLike[str]) -> TwinRegionModel:
    """Load one composed regional Twin model without recomposing its cells.

    Examples:
        region_twin = fq.twin.load_region_twin("region-twin.json")
        prediction = region_twin.predict(circuit, physical_qubits=(20, 27, 34))

    Raises:
        ValueError: If the file is unreadable, not a JSON object, not the
            supported schema version, not canonical, or no longer composes into
            a valid regional model.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin region model from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin region-model file must contain a JSON object")
    _require_fields(payload, _ARTIFACT_FIELDS, name="Twin region model")
    if payload["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported Twin region-model artifact schema")
    region_payload = payload["region"]
    twin_payload = payload["twin"]
    if not isinstance(region_payload, Mapping) or not isinstance(twin_payload, Mapping):
        raise ValueError("Twin region-model members must be JSON objects")
    region = _region_from_dict(region_payload)
    frozen_twin = _frozen_twin_from_dict(twin_payload)
    try:
        region_twin = TwinRegionModel(region=region, twin=frozen_twin)
    except _INVALID_MEMBERS as error:
        raise ValueError("Invalid Twin region model") from error
    if not _matches_canonical_payload(_to_dict(region_twin), payload):
        raise ValueError("Twin region model is not in canonical v1 form")
    return region_twin


def dump_region_twin(
    region_twin: TwinRegionModel,
    path: str | PathLike[str],
) -> None:
    """Write one private regional Twin model once; never replace another.

    Examples:
        fq.twin.dump_region_twin(region_twin, "region-twin.json")

    Raises:
        TypeError: If ``region_twin`` is not a
            :class:`~flagquantum.twin.TwinRegionModel`.
        ValueError: If the model is no longer internally consistent, or if the
            destination already holds a different or invalid artifact.
    """

    payload = _to_dict(region_twin)
    encoded = _canonical(payload) + "\n"
    write_once(
        Path(path),
        encoded,
        label="Twin region model",
        matches=lambda destination: (
            _canonical(_to_dict(load_region_twin(destination))) == _canonical(payload)
        ),
    )


__all__ = ("dump_region_twin", "load_region_twin")
