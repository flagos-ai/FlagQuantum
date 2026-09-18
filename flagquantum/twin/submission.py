"""Persistent bindings between frozen Twin experiments and remote receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..remote.qpu import DeploymentResult, ProviderTaskHandle, build_result_metadata
from ._atomic import write_once
from .experiment import TwinExperiment, TwinHardwareReport
from .prediction import TwinPrediction

_SUBMISSION_SCHEMA = "flagquantum.twin_submission.v1"
_RECEIPT_FIELDS = (
    "deployment_receipt_schema",
    "deployment_package_schema",
    "deployment_program_format",
    "routing_evidence_sha256",
    "deployment_artifact_sha256",
    "submitted_qasm_sha256",
    "compiler",
    "target_qubits",
)


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(payload: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        normalized: object = json.loads(
            json.dumps(dict(payload), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only JSON values") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a JSON object")
    return normalized


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


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


def _digest(value: str, *, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _prediction_from_dict(payload: Mapping[str, Any]) -> TwinPrediction:
    _require_fields(
        payload,
        {
            "schema",
            "snapshot_identity",
            "circuit_identity",
            "n_wires",
            "ideal_probabilities",
            "twin_probabilities",
            "total_variation_from_ideal",
        },
        name="Twin prediction",
    )
    try:
        snapshot_identity = _digest(
            str(payload["snapshot_identity"]), name="snapshot_identity"
        )
        circuit_identity = _digest(
            str(payload["circuit_identity"]), name="circuit_identity"
        )
        return TwinPrediction(
            schema=str(payload["schema"]),
            snapshot_identity=snapshot_identity,
            circuit_identity=circuit_identity,
            n_wires=int(payload["n_wires"]),
            ideal_probabilities=tuple(payload["ideal_probabilities"]),
            twin_probabilities=tuple(payload["twin_probabilities"]),
            total_variation_from_ideal=float(payload["total_variation_from_ideal"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin prediction") from error


def _experiment_from_dict(payload: Mapping[str, Any]) -> TwinExperiment:
    _require_fields(
        payload,
        {
            "schema",
            "snapshot_identity",
            "prediction",
            "name",
            "backend_name",
            "target_qubits",
            "submitted_qasm",
            "submitted_qasm_identity",
            "shots",
        },
        name="Twin experiment",
    )
    prediction_payload = payload["prediction"]
    if not isinstance(prediction_payload, Mapping):
        raise ValueError("Twin experiment prediction must be a JSON object")
    try:
        experiment = TwinExperiment(
            schema=str(payload["schema"]),
            snapshot_identity=_digest(
                str(payload["snapshot_identity"]), name="snapshot_identity"
            ),
            prediction=_prediction_from_dict(prediction_payload),
            name=str(payload["name"]),
            backend_name=str(payload["backend_name"]),
            target_qubits=tuple(payload["target_qubits"]),
            submitted_qasm=str(payload["submitted_qasm"]),
            submitted_qasm_identity=_digest(
                str(payload["submitted_qasm_identity"]),
                name="submitted_qasm_identity",
            ),
            shots=int(payload["shots"]),
        )
        if not experiment.name.strip() or not experiment.backend_name.strip():
            raise ValueError("Twin experiment requires a name and backend")
        if not experiment.submitted_qasm.lstrip().startswith("OPENQASM 2.0;"):
            raise ValueError("Twin experiment requires OpenQASM 2.0")
        if experiment.shots <= 0 or experiment.shots % 1024:
            raise ValueError(
                "Twin experiment shots must be a positive multiple of 1024"
            )
        return experiment
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Twin experiment") from error


@dataclass(frozen=True)
class TwinSubmission:
    """A frozen Twin experiment bound to one already submitted remote task."""

    experiment: TwinExperiment
    receipt: ProviderTaskHandle
    schema: str = _SUBMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUBMISSION_SCHEMA:
            raise ValueError("unsupported Twin submission schema")
        if not isinstance(self.experiment, TwinExperiment):
            raise TypeError("experiment must be a TwinExperiment")
        if not isinstance(self.receipt, ProviderTaskHandle):
            raise TypeError("receipt must be a ProviderTaskHandle")
        receipt = self.receipt
        experiment = self.experiment
        if (
            receipt.provider != "quafu"
            or receipt.backend_name != experiment.backend_name
        ):
            raise ValueError("submission receipt identifies a different QPU target")
        if not receipt.task_id.strip():
            raise ValueError("submission receipt requires a task ID")
        build_result_metadata(receipt)
        if (
            receipt.payload.get("submitted_qasm_sha256")
            != experiment.submitted_qasm_identity
        ):
            raise ValueError("submission receipt does not match the frozen QASM")
        if tuple(receipt.payload.get("target_qubits", ())) != experiment.target_qubits:
            raise ValueError("submission receipt does not match the physical mapping")
        normalized_payload = _json_mapping(
            receipt.payload, name="submission receipt payload"
        )
        retained_payload = {
            field: normalized_payload[field]
            for field in _RECEIPT_FIELDS
            if field in normalized_payload
        }
        frozen_receipt = ProviderTaskHandle(
            provider=receipt.provider,
            task_id=receipt.task_id,
            backend_name=receipt.backend_name,
            payload=_freeze_json(retained_payload),
        )
        object.__setattr__(self, "receipt", frozen_receipt)

    @classmethod
    def from_receipt(
        cls,
        experiment: TwinExperiment,
        receipt: ProviderTaskHandle,
    ) -> TwinSubmission:
        """Bind an existing receipt without submitting or contacting a provider."""

        return cls(experiment=experiment, receipt=receipt)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def validate_result(self, result: DeploymentResult) -> TwinHardwareReport:
        """Validate one fetched result against the restored experiment and receipt."""

        return self.experiment.validate_result(result, receipt=self.receipt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment": self.experiment.to_dict(),
            "receipt": {
                "provider": self.receipt.provider,
                "task_id": self.receipt.task_id,
                "backend_name": self.receipt.backend_name,
                "payload": _thaw_json(self.receipt.payload),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinSubmission:
        """Restore a submission from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin submission must be a mapping")
        _require_fields(
            payload, {"schema", "experiment", "receipt"}, name="Twin submission"
        )
        experiment_payload = payload["experiment"]
        receipt_payload = payload["receipt"]
        if not isinstance(experiment_payload, Mapping) or not isinstance(
            receipt_payload, Mapping
        ):
            raise ValueError("Twin submission members must be JSON objects")
        _require_fields(
            receipt_payload,
            {"provider", "task_id", "backend_name", "payload"},
            name="Twin submission receipt",
        )
        native_payload = receipt_payload["payload"]
        if not isinstance(native_payload, Mapping):
            raise ValueError("Twin submission receipt payload must be a JSON object")
        try:
            receipt = ProviderTaskHandle(
                provider=str(receipt_payload["provider"]),
                task_id=str(receipt_payload["task_id"]),
                backend_name=str(receipt_payload["backend_name"]),
                payload=_json_mapping(
                    native_payload, name="submission receipt payload"
                ),
            )
            return cls(
                schema=str(payload["schema"]),
                experiment=_experiment_from_dict(experiment_payload),
                receipt=receipt,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin submission") from error


def load_submission(path: str | PathLike[str]) -> TwinSubmission:
    """Load a frozen submission offline without polling or submitting a task."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin submission from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin submission file must contain a JSON object")
    return TwinSubmission.from_dict(payload)


def dump_submission(
    submission: TwinSubmission,
    path: str | PathLike[str],
) -> None:
    """Write a private submission file once; never replace different content."""

    if not isinstance(submission, TwinSubmission):
        raise TypeError("submission must be a TwinSubmission")
    encoded = (
        json.dumps(
            submission.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin submission",
        matches=lambda destination: (
            load_submission(destination).identity == submission.identity
        ),
    )


__all__ = ("TwinSubmission", "dump_submission", "load_submission")
