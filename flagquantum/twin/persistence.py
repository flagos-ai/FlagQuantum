"""Offline persistence for frozen QPU digital-twin models."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ..noise import NoiseModel
from .model import QPUDigitalTwin, TwinSnapshot

_TWIN_SCHEMA = "flagquantum.qpu_digital_twin.v1"
_TWIN_FIELDS = {"schema", "snapshot", "noise_model"}
_SNAPSHOT_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "captured_at",
    "physical_qubits",
    "calibration_identity",
    "noise_model_identity",
}


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _require_fields(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} fields do not match the v1 schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _snapshot_from_dict(payload: Mapping[str, Any]) -> TwinSnapshot:
    _require_fields(payload, _SNAPSHOT_FIELDS, name="Twin snapshot")
    try:
        snapshot = TwinSnapshot(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            captured_at=str(payload["captured_at"]),
            physical_qubits=tuple(int(value) for value in payload["physical_qubits"]),
            calibration_identity=str(payload["calibration_identity"]),
            noise_model_identity=str(payload["noise_model_identity"]),
        )
        if _canonical(snapshot.to_dict()) != _canonical(payload):
            raise ValueError("Twin snapshot is not in canonical v1 form")
        return snapshot
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin snapshot") from error


def _to_dict(twin: QPUDigitalTwin) -> dict[str, Any]:
    if not isinstance(twin, QPUDigitalTwin):
        raise TypeError("twin must be a QPUDigitalTwin")
    if twin.noise_model.identity != twin.snapshot.noise_model_identity:
        raise ValueError("Twin noise model changed after the snapshot was frozen")
    profile = twin.noise_model.device_profile
    if profile is None or profile.identity != twin.snapshot.calibration_identity:
        raise ValueError("Twin calibration changed after the snapshot was frozen")
    return {
        "schema": _TWIN_SCHEMA,
        "snapshot": twin.snapshot.to_dict(),
        "noise_model": twin.noise_model.to_dict(),
    }


def load_twin(path: str | PathLike[str]) -> QPUDigitalTwin:
    """Load one frozen Twin offline without contacting a provider."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load QPU digital Twin from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("QPU digital Twin file must contain a JSON object")
    _require_fields(payload, _TWIN_FIELDS, name="QPU digital Twin")
    if payload["schema"] != _TWIN_SCHEMA:
        raise ValueError("unsupported QPU digital Twin schema")
    snapshot_payload = payload["snapshot"]
    noise_payload = payload["noise_model"]
    if not isinstance(snapshot_payload, Mapping) or not isinstance(
        noise_payload, Mapping
    ):
        raise ValueError("QPU digital Twin members must be JSON objects")
    try:
        model = NoiseModel.from_dict(noise_payload)
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        IndexError,
    ) as error:
        raise ValueError("Invalid QPU digital Twin noise model") from error
    try:
        canonical_model = _canonical(model.to_dict())
        canonical_payload = _canonical(noise_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid QPU digital Twin noise model") from error
    if canonical_model != canonical_payload:
        raise ValueError("noise model is not in canonical v1 form")
    try:
        return QPUDigitalTwin(
            snapshot=_snapshot_from_dict(snapshot_payload),
            noise_model=model,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid QPU digital Twin") from error


def dump_twin(twin: QPUDigitalTwin, path: str | PathLike[str]) -> None:
    """Write one private Twin file once; never replace a different model."""

    payload = _to_dict(twin)
    destination = Path(path)
    encoded = _canonical(payload) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing_payload = _to_dict(load_twin(destination))
        except ValueError as error:
            raise ValueError(
                f"Refusing to replace invalid QPU digital Twin at {destination}"
            ) from error
        if _identity(existing_payload) != _identity(payload):
            raise ValueError(
                f"Refusing to replace different QPU digital Twin at {destination}"
            )
    except OSError as error:
        raise ValueError(f"Cannot write QPU digital Twin to {destination}") from error


__all__ = ("dump_twin", "load_twin")
