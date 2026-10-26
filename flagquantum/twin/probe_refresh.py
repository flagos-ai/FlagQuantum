"""Deterministic, offline planning for regional Twin evidence refreshes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.ir import CircuitIR, ensure_circuit_ir
from .history import TwinCalibrationHistory
from .region import TwinConnectedRegion, TwinRegionCoverage
from .region_model import TwinRegionModel
from .region_release import TwinRegionRelease
from .region_suite import TwinRegionValidationSuite, prepare_region_validation_suite
from .region_validation_history import TwinRegionValidationHistory
from .validation_history import TwinValidationHistory

_PLAN_SCHEMA = "flagquantum.twin_probe_refresh_plan.v1"
_REPETITIONS = 2


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _history_identity(history: Any) -> str:
    return _identity(history.to_dict())


def _release_region(release: TwinRegionRelease) -> TwinConnectedRegion:
    return TwinConnectedRegion(
        provider=release.provider,
        backend_name=release.backend_name,
        captured_at=release.candidate_captured_at,
        physical_qubits=release.physical_qubits,
        directed_couplers=release.directed_couplers,
        supported_operations=release.supported_operations,
        maximum_instruction_count=release.maximum_instruction_count,
        maximum_circuit_depth=release.maximum_circuit_depth,
        source_snapshot_identities=release.source_snapshot_identities,
        source_support_identities=release.source_support_identities,
    )


def _validate_calibration_history(
    release: TwinRegionRelease,
    history: TwinCalibrationHistory | None,
) -> None:
    if history is None:
        return
    if not isinstance(history, TwinCalibrationHistory):
        raise TypeError("calibration_history must be a TwinCalibrationHistory")
    if (
        history.provider,
        history.backend_name,
        history.physical_qubits,
    ) != (release.provider, release.backend_name, release.physical_qubits):
        raise ValueError("calibration history does not match the released region")
    if release.candidate_snapshot_identity not in history.snapshot_identities:
        raise ValueError("calibration history is not anchored to the release snapshot")


def _validation_circuits(
    release: TwinRegionRelease,
    history: TwinValidationHistory | TwinRegionValidationHistory | None,
) -> tuple[str, ...]:
    if history is None:
        return ()
    if not isinstance(history, (TwinValidationHistory, TwinRegionValidationHistory)):
        raise TypeError(
            "validation_history must be a TwinValidationHistory or "
            "TwinRegionValidationHistory"
        )
    if (
        history.provider,
        history.backend_name,
        history.physical_qubits,
    ) != (release.provider, release.backend_name, release.physical_qubits):
        raise ValueError("validation history does not match the released region")
    if isinstance(history, TwinRegionValidationHistory):
        return history.circuit_identities
    return (history.circuit_identity,)


def _used_scopes(
    ir: CircuitIR, mapping: tuple[int, ...]
) -> tuple[set[int], set[tuple[str, tuple[int, ...]]]]:
    qubits: set[int] = set()
    operations: set[tuple[str, tuple[int, ...]]] = set()
    for instruction in ir.instructions:
        physical_scope = tuple(mapping[wire] for wire in instruction.wires)
        qubits.update(physical_scope)
        operations.add((instruction.name, physical_scope))
    return qubits, operations


def _drift_overlap(
    ir: CircuitIR,
    mapping: tuple[int, ...],
    history: TwinCalibrationHistory | None,
) -> int:
    if history is None:
        return 0
    drift = history.latest_interval_drift
    used_qubits, used_operations = _used_scopes(ir, mapping)
    score = int(drift.channel_model_changed)
    for qubit_drift in drift.qubit_drifts:
        changed = (
            qubit_drift.relative_t1_delta != 0.0
            or qubit_drift.relative_t2_delta != 0.0
            or (qubit_drift.maximum_readout_row_tv_distance or 0.0) != 0.0
        )
        if changed and qubit_drift.physical_qubit in used_qubits:
            score += 1
    for gate_drift in drift.gate_duration_drifts:
        changed = (
            gate_drift.absolute_duration_delta_seconds != 0.0
            or (gate_drift.relative_duration_delta or 0.0) != 0.0
        )
        if not changed:
            continue
        scope = gate_drift.physical_qubits
        if scope is None:
            if any(name == gate_drift.gate_name for name, _ in used_operations):
                score += 1
        elif (gate_drift.gate_name, scope) in used_operations:
            score += 1
    return score


@dataclass(frozen=True)
class TwinProbeRefreshPlan:
    """Immutable selection of caller-supplied regional validation probes.

    The plan is a preparation artifact. It does not contain credentials,
    submit work, poll tasks, refresh calibration, or modify a Twin.
    """

    release_identity: str
    candidate_region_identity: str
    candidate_snapshot_identity: str
    target: str
    physical_qubits: tuple[int, ...]
    candidate_circuit_identities: tuple[str, ...]
    selected_circuit_identities: tuple[str, ...]
    unsupported_candidates: tuple[tuple[str, tuple[str, ...]], ...]
    covered_directed_couplers: tuple[tuple[int, int], ...]
    coverage_gaps: tuple[str, ...]
    calibration_history_identity: str | None
    validation_history_identity: str | None
    shots: int
    maximum_tasks: int
    repetitions: int = _REPETITIONS
    schema: str = _PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PLAN_SCHEMA:
            raise ValueError("unsupported Twin probe-refresh plan schema")
        for name in (
            "release_identity",
            "candidate_region_identity",
            "candidate_snapshot_identity",
        ):
            _digest(str(getattr(self, name)), name)
        for name in ("calibration_history_identity", "validation_history_identity"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if not self.target.strip() or ":" not in self.target:
            raise ValueError("target must use provider:backend form")
        mapping = tuple(self.physical_qubits)
        if (
            not mapping
            or len(mapping) != len(set(mapping))
            or any(type(qubit) is not int or qubit < 0 for qubit in mapping)
        ):
            raise ValueError("physical_qubits must be unique non-negative integers")
        candidates = tuple(self.candidate_circuit_identities)
        selected = tuple(self.selected_circuit_identities)
        for identity in (*candidates, *selected):
            _digest(identity, "circuit_identity")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate circuit identities must be unique")
        if len(selected) != len(set(selected)) or any(
            identity not in candidates for identity in selected
        ):
            raise ValueError("selected circuits must be unique plan candidates")
        unsupported = tuple(
            (identity, tuple(reasons))
            for identity, reasons in self.unsupported_candidates
        )
        unsupported_ids = tuple(identity for identity, _ in unsupported)
        if len(unsupported_ids) != len(set(unsupported_ids)) or any(
            identity not in candidates for identity in unsupported_ids
        ):
            raise ValueError("unsupported circuits must be unique plan candidates")
        if set(unsupported_ids) & set(selected):
            raise ValueError("unsupported circuits cannot be selected")
        for identity, reasons in unsupported:
            _digest(identity, "unsupported_circuit_identity")
            if not reasons or any(not reason.strip() for reason in reasons):
                raise ValueError("unsupported candidates require refusal reasons")
        couplers = tuple(self.covered_directed_couplers)
        physical = set(mapping)
        if len(couplers) != len(set(couplers)) or any(
            source == target or source not in physical or target not in physical
            for source, target in couplers
        ):
            raise ValueError("covered couplers must be unique and use plan qubits")
        gaps = tuple(dict.fromkeys(str(gap) for gap in self.coverage_gaps))
        if any(not gap.strip() for gap in gaps):
            raise ValueError("coverage gaps must be non-empty strings")
        if type(self.shots) is not int or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        if type(self.maximum_tasks) is not int or self.maximum_tasks < 0:
            raise ValueError("maximum_tasks must be a non-negative integer")
        if self.repetitions != _REPETITIONS:
            raise ValueError("probe refresh plans require exactly two repetitions")
        if len(selected) * self.repetitions > self.maximum_tasks:
            raise ValueError("planned task count exceeds maximum_tasks")
        object.__setattr__(self, "physical_qubits", mapping)
        object.__setattr__(self, "candidate_circuit_identities", candidates)
        object.__setattr__(self, "selected_circuit_identities", selected)
        object.__setattr__(self, "unsupported_candidates", unsupported)
        object.__setattr__(self, "covered_directed_couplers", couplers)
        object.__setattr__(self, "coverage_gaps", gaps)

    @property
    def identity(self) -> str:
        """Return the deterministic plan identity."""

        return _identity(self.to_dict())

    @property
    def planned_task_count(self) -> int:
        """Return the exact number of separately submitted tasks in the plan."""

        return len(self.selected_circuit_identities) * self.repetitions

    @property
    def planned_shots(self) -> int:
        """Return the total shots across every planned task."""

        return self.planned_task_count * self.shots

    def prepare_suite(
        self,
        region_twin: TwinRegionModel,
        candidate_circuits: Sequence[Any],
        *,
        name: str,
    ) -> TwinRegionValidationSuite:
        """Materialize the selected probes through the existing suite path.

        This method remains offline. The returned suite still requires an
        explicit provider submission for every frozen experiment.
        """

        if not isinstance(region_twin, TwinRegionModel):
            raise TypeError("region_twin must be a TwinRegionModel")
        if (
            region_twin.identity != self.candidate_region_identity
            or region_twin.twin.snapshot.identity != self.candidate_snapshot_identity
            or region_twin.target != self.target
            or region_twin.region.physical_qubits != self.physical_qubits
        ):
            raise ValueError("regional Twin does not match the probe-refresh plan")
        indexed: dict[str, Any] = {}
        for circuit in tuple(candidate_circuits):
            identity = ensure_circuit_ir(circuit).content_hash
            if identity in indexed:
                raise ValueError("candidate circuits must have unique identities")
            indexed[identity] = circuit
        missing = tuple(
            identity
            for identity in self.selected_circuit_identities
            if identity not in indexed
        )
        if missing:
            raise ValueError("candidate circuits do not contain every selected probe")
        selected = tuple(
            indexed[identity] for identity in self.selected_circuit_identities
        )
        if len(selected) < 2:
            raise ValueError(
                "at least two selected probes are required to prepare a regional suite"
            )
        suite = prepare_region_validation_suite(
            region_twin,
            selected,
            physical_qubits=self.physical_qubits,
            name=name,
            shots=self.shots,
            repetitions=self.repetitions,
        )
        if suite.planned_task_count != self.planned_task_count:
            raise RuntimeError("prepared suite does not match the frozen task count")
        return suite

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation of the plan."""

        return {
            "schema": self.schema,
            "release_identity": self.release_identity,
            "candidate_region_identity": self.candidate_region_identity,
            "candidate_snapshot_identity": self.candidate_snapshot_identity,
            "target": self.target,
            "physical_qubits": list(self.physical_qubits),
            "candidate_circuit_identities": list(self.candidate_circuit_identities),
            "selected_circuit_identities": list(self.selected_circuit_identities),
            "unsupported_candidates": [
                {"circuit_identity": identity, "reasons": list(reasons)}
                for identity, reasons in self.unsupported_candidates
            ],
            "covered_directed_couplers": [
                list(coupler) for coupler in self.covered_directed_couplers
            ],
            "coverage_gaps": list(self.coverage_gaps),
            "calibration_history_identity": self.calibration_history_identity,
            "validation_history_identity": self.validation_history_identity,
            "shots": self.shots,
            "maximum_tasks": self.maximum_tasks,
            "repetitions": self.repetitions,
            "planned_task_count": self.planned_task_count,
            "planned_shots": self.planned_shots,
        }


@dataclass(frozen=True)
class _Candidate:
    ir: CircuitIR
    coverage: TwinRegionCoverage
    priority: tuple[int, int, int, int, int]


def prepare_probe_refresh(
    release: TwinRegionRelease,
    candidate_circuits: Sequence[Any],
    *,
    calibration_history: TwinCalibrationHistory | None = None,
    validation_history: (
        TwinValidationHistory | TwinRegionValidationHistory | None
    ) = None,
    maximum_tasks: int,
    shots: int,
) -> TwinProbeRefreshPlan:
    """Freeze a bounded refresh plan from caller-supplied circuits only.

    Ranking first prefers structurally covered circuits absent from the exact
    release, then circuits absent from validation history, circuits overlapping
    observed interval drift, wider directed-coupler coverage, and instruction
    count. Circuit identity breaks all ties, so equivalent inputs produce the
    same plan regardless of candidate order.

    Examples:
        plan = fq.twin.prepare_probe_refresh(
            release,
            candidate_circuits,
            calibration_history=calibration_history,
            validation_history=validation_history,
            maximum_tasks=12,
            shots=1024,
        )
        print(plan.selected_circuit_identities)
        print(plan.planned_task_count)
    """

    if not isinstance(release, TwinRegionRelease):
        raise TypeError("release must be a TwinRegionRelease")
    if isinstance(candidate_circuits, (str, bytes)):
        raise TypeError("candidate_circuits must be a sequence of circuits")
    if type(maximum_tasks) is not int or maximum_tasks < 0:
        raise ValueError("maximum_tasks must be a non-negative integer")
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    _validate_calibration_history(release, calibration_history)
    validated = set(_validation_circuits(release, validation_history))
    frozen_candidates = tuple(candidate_circuits)
    region = _release_region(release)
    mapping = release.physical_qubits
    eligible: list[_Candidate] = []
    unsupported: list[tuple[str, tuple[str, ...]]] = []
    identities: list[str] = []
    released = set(release.verified_circuit_identities)
    for circuit in frozen_candidates:
        ir = ensure_circuit_ir(circuit)
        identity = ir.content_hash
        identities.append(identity)
        if ir.n_wires != len(mapping):
            unsupported.append((identity, ("physical_mapping_mismatch",)))
            continue
        coverage = region.coverage_report(circuit, physical_qubits=mapping)
        if coverage.status != "covered":
            unsupported.append((identity, coverage.reasons))
            continue
        eligible.append(
            _Candidate(
                ir=ir,
                coverage=coverage,
                priority=(
                    int(identity not in released),
                    int(identity not in validated),
                    _drift_overlap(ir, mapping, calibration_history),
                    len(coverage.required_directed_couplers),
                    len(ir.instructions),
                ),
            )
        )
    if len(identities) != len(set(identities)):
        raise ValueError("candidate circuits must have unique identities")

    eligible.sort(
        key=lambda item: (
            *(-value for value in item.priority),
            item.ir.content_hash,
        )
    )
    capacity = maximum_tasks // _REPETITIONS
    selected = tuple(eligible[:capacity])
    selected_identities = tuple(item.ir.content_hash for item in selected)
    covered_couplers = tuple(
        dict.fromkeys(
            coupler
            for item in selected
            for coupler in item.coverage.required_directed_couplers
        )
    )

    gaps: list[str] = []
    if calibration_history is None:
        gaps.append("calibration_history_missing")
    else:
        if calibration_history.snapshot_identities[-1] != (
            release.candidate_snapshot_identity
        ):
            gaps.append("calibration_changed_since_release")
        if calibration_history.latest_interval_drift.has_observed_drift:
            gaps.append("calibration_drift_observed")
    if validation_history is None:
        gaps.append("validation_history_missing")
    elif validation_history.snapshot_identities[-1] != (
        release.candidate_snapshot_identity
    ):
        gaps.append("release_snapshot_not_latest_validation")
    if any(item.ir.content_hash not in released for item in eligible):
        gaps.append("candidate_circuits_not_in_release")
    if unsupported:
        gaps.append("unsupported_candidate_circuits")
    if len(eligible) > capacity:
        gaps.append("task_budget_exhausted")
    if len(selected) < 2:
        gaps.append("insufficient_supported_circuits_for_suite")

    return TwinProbeRefreshPlan(
        release_identity=release.identity,
        candidate_region_identity=release.candidate_region_identity,
        candidate_snapshot_identity=release.candidate_snapshot_identity,
        target=release.target,
        physical_qubits=mapping,
        candidate_circuit_identities=tuple(sorted(identities)),
        selected_circuit_identities=selected_identities,
        unsupported_candidates=tuple(sorted(unsupported)),
        covered_directed_couplers=covered_couplers,
        coverage_gaps=tuple(gaps),
        calibration_history_identity=(
            None
            if calibration_history is None
            else _history_identity(calibration_history)
        ),
        validation_history_identity=(
            None
            if validation_history is None
            else _history_identity(validation_history)
        ),
        shots=shots,
        maximum_tasks=maximum_tasks,
    )


__all__ = ("prepare_probe_refresh", "TwinProbeRefreshPlan")
