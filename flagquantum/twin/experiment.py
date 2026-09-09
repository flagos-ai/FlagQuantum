"""Identity-bound hardware experiments for QPU digital twins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    validate_deployment_result,
)
from .model import QPUDigitalTwin
from .prediction import TwinPrediction
from .validation import TwinValidationReport

_EXPERIMENT_SCHEMA = "flagquantum.twin_experiment.v1"
_HARDWARE_REPORT_SCHEMA = "flagquantum.twin_hardware_report.v1"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class TwinExperiment:
    """A prediction frozen together with the exact program submitted to a QPU."""

    snapshot_identity: str
    prediction: TwinPrediction
    name: str
    backend_name: str
    submitted_qasm: str
    submitted_qasm_identity: str
    shots: int
    schema: str = _EXPERIMENT_SCHEMA

    @classmethod
    def prepare(
        cls,
        twin: QPUDigitalTwin,
        circuit: Any,
        *,
        submitted_qasm: str,
        name: str,
        shots: int,
    ) -> "TwinExperiment":
        """Freeze the model, prediction, and submitted program before dispatch."""

        program = str(submitted_qasm)
        task_name = str(name).strip()
        shot_count = int(shots)
        if not program.lstrip().startswith("OPENQASM 2.0;"):
            raise ValueError("twin hardware experiments require OpenQASM 2.0")
        if not task_name:
            raise ValueError("twin hardware experiment name cannot be empty")
        if shot_count <= 0 or shot_count % 1024:
            raise ValueError("Quafu shots must be a positive multiple of 1024")
        prediction = twin.predict(circuit)
        if prediction.snapshot_identity != twin.snapshot.identity:
            raise RuntimeError("prediction does not match the frozen twin snapshot")
        return cls(
            snapshot_identity=twin.snapshot.identity,
            prediction=prediction,
            name=task_name,
            backend_name=twin.snapshot.backend_name,
            submitted_qasm=program,
            submitted_qasm_identity=_sha256(program),
            shots=shot_count,
        )

    def __post_init__(self) -> None:
        if self.schema != _EXPERIMENT_SCHEMA:
            raise ValueError("unsupported twin experiment schema")
        if self.prediction.snapshot_identity != self.snapshot_identity:
            raise ValueError("experiment prediction does not match its snapshot")
        if _sha256(self.submitted_qasm) != self.submitted_qasm_identity:
            raise ValueError("submitted_qasm_identity does not match submitted_qasm")

    @property
    def identity(self) -> str:
        return _payload_identity(self.to_dict())

    def submit(self, provider: Any) -> ProviderTaskHandle:
        """Submit the frozen program and verify the returned receipt."""

        if getattr(provider, "provider", None) != "quafu" or not hasattr(
            provider, "submit_qasm"
        ):
            raise TypeError("TwinExperiment.submit requires a Quafu provider")
        handle = provider.submit_qasm(
            self.submitted_qasm,
            chip=self.backend_name,
            name=self.name,
            shots=self.shots,
        )
        if handle.provider != "quafu" or handle.backend_name != self.backend_name:
            raise RuntimeError("provider receipt identifies a different QPU target")
        if handle.payload.get("submitted_qasm_sha256") != self.submitted_qasm_identity:
            raise RuntimeError("provider receipt does not match the submitted QASM")
        return handle

    def validate_result(
        self,
        result: DeploymentResult,
        *,
        receipt: ProviderTaskHandle,
    ) -> "TwinHardwareReport":
        """Bind a remote result to this experiment and evaluate its prediction."""

        validate_deployment_result(result)
        handle = result.handle
        if handle != receipt:
            raise RuntimeError(
                "provider result does not match the submitted task receipt"
            )
        if handle.provider != "quafu" or handle.backend_name != self.backend_name:
            raise RuntimeError("provider result identifies a different QPU target")
        if handle.payload.get("submitted_qasm_sha256") != self.submitted_qasm_identity:
            raise RuntimeError("provider result does not match the submitted QASM")
        if result.shots != self.shots:
            raise RuntimeError(
                "provider result shot count does not match the experiment"
            )

        metadata = _json_mapping(result.metadata)
        reported_backend = metadata.get("chip")
        if (
            isinstance(reported_backend, str)
            and reported_backend.strip()
            and reported_backend != self.backend_name
        ):
            raise RuntimeError("provider result identifies a different QPU target")
        executed_qasm = metadata.get("transpiled")
        if not isinstance(executed_qasm, str) or not executed_qasm.strip():
            executed_qasm = metadata.get("circuit")
        if not isinstance(executed_qasm, str) or not executed_qasm.strip():
            executed_qasm = None
        executed_identity = None if executed_qasm is None else _sha256(executed_qasm)
        counts = {str(key): int(value) for key, value in result.counts.items()}
        return TwinHardwareReport(
            experiment_identity=self.identity,
            provider=handle.provider,
            backend_name=handle.backend_name,
            task_id=handle.task_id,
            submitted_qasm_identity=self.submitted_qasm_identity,
            executed_qasm_identity=executed_identity,
            counts=counts,
            validation=self.prediction.compare_counts(counts),
            result_metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "prediction": self.prediction.to_dict(),
            "name": self.name,
            "backend_name": self.backend_name,
            "submitted_qasm": self.submitted_qasm,
            "submitted_qasm_identity": self.submitted_qasm_identity,
            "shots": self.shots,
        }


@dataclass(frozen=True)
class TwinHardwareReport:
    """Post-execution diagnostic bound to a twin experiment and remote task."""

    experiment_identity: str
    provider: str
    backend_name: str
    task_id: str
    submitted_qasm_identity: str
    executed_qasm_identity: str | None
    counts: Mapping[str, int]
    validation: TwinValidationReport
    result_metadata: Mapping[str, Any]
    schema: str = _HARDWARE_REPORT_SCHEMA

    @property
    def executed_program_matches_submission(self) -> bool:
        """Whether the program reported after execution matches the submission."""

        return self.executed_qasm_identity == self.submitted_qasm_identity

    @property
    def validation_scope(self) -> str:
        """Describe what the current Quafu evidence can establish."""

        return "retrospective_diagnostic"

    @property
    def identity(self) -> str:
        return _payload_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_identity": self.experiment_identity,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "task_id": self.task_id,
            "submitted_qasm_identity": self.submitted_qasm_identity,
            "executed_qasm_identity": self.executed_qasm_identity,
            "executed_program_matches_submission": (
                self.executed_program_matches_submission
            ),
            "validation_scope": self.validation_scope,
            "counts": dict(sorted(self.counts.items())),
            "validation": self.validation.to_dict(),
            "result_metadata": _json_mapping(self.result_metadata),
        }


__all__ = ("TwinExperiment", "TwinHardwareReport")
