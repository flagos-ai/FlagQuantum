"""Identity-bound hardware experiments for QPU digital twins."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..compiler.openqasm import emit_openqasm
from ..core.ir import ensure_circuit_ir
from ..remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    validate_deployment_result,
)
from ._statistics import finite_shot_tv_radius
from .evidence import TwinEvidenceEnvelope
from .model import QPUDigitalTwin
from .prediction import TwinPrediction
from .validation import TwinValidationReport

if TYPE_CHECKING:
    from .series import TwinValidationSeries

_EXPERIMENT_SCHEMA = "flagquantum.twin_experiment.v1"
_HARDWARE_REPORT_SCHEMA = "flagquantum.twin_hardware_report.v1"
_QASM_QUBIT_DECLARATION = re.compile(r"qreg\s+q\[([1-9][0-9]*)\]\s*;")
_QASM_CLASSICAL_DECLARATION = re.compile(r"creg\s+c\[([1-9][0-9]*)\]\s*;")
_QASM_QUBIT_REFERENCE = re.compile(r"q\[([0-9]+)\]")
_QASM_MEASUREMENT = re.compile(r"measure\s+q\[([0-9]+)\]\s*->\s*c\[([0-9]+)\]\s*;")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(payload), sort_keys=True, allow_nan=False)
    normalized: object = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise TypeError("normalized experiment metadata must be a JSON object")
    return normalized


def _provider_transpilation_preserves_mapping(
    executed_qasm: str,
    *,
    target_qubits: tuple[int, ...],
) -> bool:
    """Check the observable physical footprint of provider-transpiled QASM."""

    lines = tuple(line.strip() for line in executed_qasm.splitlines() if line.strip())
    if len(lines) < 5 or lines[:2] != (
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
    ):
        return False
    declarations = tuple(
        match
        for line in lines
        if (match := _QASM_QUBIT_DECLARATION.fullmatch(line)) is not None
    )
    if len(declarations) != 1:
        return False
    register_width = int(declarations[0].group(1))
    if any(qubit >= register_width for qubit in target_qubits):
        return False
    classical_declarations = tuple(
        match
        for line in lines
        if (match := _QASM_CLASSICAL_DECLARATION.fullmatch(line)) is not None
    )
    if len(classical_declarations) != 1 or int(
        classical_declarations[0].group(1)
    ) != len(target_qubits):
        return False

    body = tuple(
        line
        for line in lines
        if _QASM_QUBIT_DECLARATION.fullmatch(line) is None
        and _QASM_CLASSICAL_DECLARATION.fullmatch(line) is None
        and line not in {"OPENQASM 2.0;", 'include "qelib1.inc";'}
    )
    referenced = {
        int(match.group(1))
        for line in body
        for match in _QASM_QUBIT_REFERENCE.finditer(line)
    }
    if referenced != set(target_qubits):
        return False

    measurements = []
    for line in body:
        if not line.startswith("measure"):
            continue
        match = _QASM_MEASUREMENT.fullmatch(line)
        if match is None:
            return False
        measurements.append((int(match.group(2)), int(match.group(1))))
    ordered_measurements = tuple(sorted(measurements))
    return ordered_measurements == tuple(enumerate(target_qubits))


@dataclass(frozen=True)
class TwinExperiment:
    """A prediction frozen together with the exact program submitted to a QPU."""

    snapshot_identity: str
    prediction: TwinPrediction
    name: str
    backend_name: str
    target_qubits: tuple[int, ...]
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
        submitted_qasm: str | None = None,
        name: str,
        shots: int,
    ) -> TwinExperiment:
        """Freeze the model, prediction, and submitted program before dispatch.

        Omitting ``submitted_qasm`` emits deterministic OpenQASM 2.0 directly
        from the FlagQuantum circuit. That direct binding is required when a
        later hardware report is converted into verified Twin evidence.
        """

        program = (
            emit_openqasm(circuit, version=2.0)
            if submitted_qasm is None
            else str(submitted_qasm)
        )
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
            target_qubits=twin.snapshot.physical_qubits,
            submitted_qasm=program,
            submitted_qasm_identity=_sha256(program),
            shots=shot_count,
        )

    def __post_init__(self) -> None:
        if self.schema != _EXPERIMENT_SCHEMA:
            raise ValueError("unsupported twin experiment schema")
        if self.prediction.snapshot_identity != self.snapshot_identity:
            raise ValueError("experiment prediction does not match its snapshot")
        if (
            len(self.target_qubits) != self.prediction.n_wires
            or len(set(self.target_qubits)) != len(self.target_qubits)
            or any(qubit < 0 for qubit in self.target_qubits)
        ):
            raise ValueError(
                "target_qubits must map each logical wire to one physical qubit"
            )
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
            target_qubits=self.target_qubits,
        )
        if not isinstance(handle, ProviderTaskHandle):
            raise TypeError("Quafu provider must return a ProviderTaskHandle.")
        if handle.provider != "quafu" or handle.backend_name != self.backend_name:
            raise RuntimeError("provider receipt identifies a different QPU target")
        if handle.payload.get("submitted_qasm_sha256") != self.submitted_qasm_identity:
            raise RuntimeError("provider receipt does not match the submitted QASM")
        if tuple(handle.payload.get("target_qubits", ())) != self.target_qubits:
            raise RuntimeError("provider receipt does not match the physical mapping")
        return handle

    def validate_result(
        self,
        result: DeploymentResult,
        *,
        receipt: ProviderTaskHandle,
    ) -> TwinHardwareReport:
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
        if tuple(handle.payload.get("target_qubits", ())) != self.target_qubits:
            raise RuntimeError("provider result does not match the physical mapping")
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

    def evidence_from_report(
        self,
        report: TwinHardwareReport,
        *,
        circuit: Any,
        confidence_level: float = 0.95,
    ) -> TwinEvidenceEnvelope:
        """Convert one directly bound hardware report into exact evidence.

        The method fails closed unless the FlagQuantum circuit, canonical
        submitted OpenQASM, provider result, and reported executed program all
        retain the identities frozen by this experiment.
        """

        if not isinstance(report, TwinHardwareReport):
            raise TypeError("report must be a TwinHardwareReport")
        if report.experiment_identity != self.identity:
            raise RuntimeError("hardware report does not match this experiment")
        if report.provider != "quafu" or report.backend_name != self.backend_name:
            raise RuntimeError("hardware report identifies a different QPU target")
        executed_qasm = report.result_metadata.get("transpiled")
        if (
            not isinstance(executed_qasm, str)
            or not executed_qasm.strip()
            or _sha256(executed_qasm) != report.executed_qasm_identity
        ):
            raise RuntimeError("authoritative executed program is unavailable")

        ir = ensure_circuit_ir(circuit)
        if ir.content_hash != self.prediction.circuit_identity:
            raise RuntimeError("circuit does not match the frozen prediction")
        canonical_qasm = emit_openqasm(circuit, version=2.0)
        if self.submitted_qasm != canonical_qasm:
            raise RuntimeError(
                "submitted QASM is not directly bound to the FlagQuantum circuit"
            )
        if not report.executed_program_matches_submission:
            echoed_qasm = report.result_metadata.get("circuit")
            if echoed_qasm != self.submitted_qasm:
                raise RuntimeError(
                    "provider transpilation is not bound to the frozen submission"
                )
            if not _provider_transpilation_preserves_mapping(
                executed_qasm,
                target_qubits=self.target_qubits,
            ):
                raise RuntimeError(
                    "provider transpilation changes the frozen physical mapping"
                )
        expected_validation = self.prediction.compare_counts(report.counts)
        if (
            report.validation != expected_validation
            or expected_validation.shots != self.shots
        ):
            raise RuntimeError("hardware validation does not match the frozen result")

        finite_shot_radius = finite_shot_tv_radius(
            outcome_count=2**self.prediction.n_wires,
            shots=expected_validation.shots,
            confidence_level=confidence_level,
        )
        verified_radius = min(
            1.0,
            expected_validation.twin_hardware_total_variation + finite_shot_radius,
        )
        operations = tuple(
            sorted({instruction.name for instruction in ir.instructions} | {"measure"})
        )
        return TwinEvidenceEnvelope(
            snapshot_identity=self.snapshot_identity,
            physical_qubits=self.target_qubits,
            supported_operations=operations,
            maximum_instruction_count=len(ir.instructions),
            verified_circuit_identities=(ir.content_hash,),
            evidence_identity=report.identity,
            verified_tv_error_bound=verified_radius,
            estimated_tv_error_bound=None,
            confidence_level=confidence_level,
        )

    def validation_series(
        self,
        reports: Sequence[TwinHardwareReport],
        *,
        circuit: Any,
        confidence_level: float = 0.95,
    ) -> TwinValidationSeries:
        """Summarize distinct bound results with one simultaneous confidence.

        Examples:
            series = experiment.validation_series(
                [first_report, second_report],
                circuit=circuit,
            )
            evidence = series.to_evidence()
        """

        from .series import TwinValidationSeries

        return TwinValidationSeries._from_reports(
            self,
            reports,
            circuit=circuit,
            confidence_level=confidence_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "prediction": self.prediction.to_dict(),
            "name": self.name,
            "backend_name": self.backend_name,
            "target_qubits": list(self.target_qubits),
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
