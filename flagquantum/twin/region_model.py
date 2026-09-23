"""Fail-closed composition of compatible local Twin noise models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.ir import ensure_circuit_ir
from ..noise import (
    CorrelatedReadoutError,
    DeviceNoiseProfile,
    GateDuration,
    KrausChannel,
    NoiseModel,
    QubitNoiseCalibration,
    ReadoutError,
)
from .circuit_support import TwinCircuitSupport
from .experiment import TwinExperiment
from .factory import from_noise_model
from .model import QPUDigitalTwin
from .prediction import TwinPrediction
from .region import TwinConnectedRegion, compose_connected_region

if TYPE_CHECKING:
    from .series import TwinValidationSeries

_REGION_MODEL_SCHEMA = "flagquantum.twin_region_model.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _region_profile_source(region_identity: str) -> str:
    """Return the device-profile provenance bound to one composed region."""

    return f"flagquantum:twin-region:{region_identity}"


def _same_qubit_calibration(
    left: QubitNoiseCalibration,
    right: QubitNoiseCalibration,
) -> bool:
    return (
        left.t1 == right.t1
        and left.t2 == right.t2
        and left.excited_population == right.excited_population
        and left.readout_error == right.readout_error
    )


def _remap_wires(
    wires: tuple[int, ...],
    *,
    local_to_region: Mapping[int, int],
) -> tuple[int, ...]:
    try:
        return tuple(local_to_region[wire] for wire in wires)
    except KeyError as error:
        raise ValueError(
            "Twin noise rule references a wire outside its mapping"
        ) from error


def _compose_device_profile(
    cells: tuple[tuple[QPUDigitalTwin, TwinCircuitSupport], ...],
    region: TwinConnectedRegion,
) -> DeviceNoiseProfile:
    region_wire = {
        physical_qubit: wire
        for wire, physical_qubit in enumerate(region.physical_qubits)
    }
    calibrations: dict[int, QubitNoiseCalibration] = {}
    durations: dict[tuple[str, tuple[int, ...] | None], GateDuration] = {}
    time_unit: str | None = None

    for twin, _ in cells:
        profile = twin.noise_model.device_profile
        if profile is None:
            raise ValueError("every Twin region cell requires a device noise profile")
        if time_unit is None:
            time_unit = profile.time_unit
        elif profile.time_unit != time_unit:
            raise ValueError("Twin region profiles must use one calibration time unit")
        local_to_region = {
            local: region_wire[physical]
            for local, physical in enumerate(twin.snapshot.physical_qubits)
        }
        for calibration in profile.qubits:
            if calibration.wire not in local_to_region:
                raise ValueError(
                    "Twin device profile references a wire outside its mapping"
                )
            regional_wire = local_to_region[calibration.wire]
            remapped_calibration = QubitNoiseCalibration(
                wire=regional_wire,
                t1=calibration.t1,
                t2=calibration.t2,
                excited_population=calibration.excited_population,
                readout_error=calibration.readout_error,
            )
            existing_calibration = calibrations.get(regional_wire)
            if existing_calibration is not None and not _same_qubit_calibration(
                existing_calibration, remapped_calibration
            ):
                raise ValueError("overlapping Twin cells disagree on qubit calibration")
            calibrations[regional_wire] = remapped_calibration
        for duration in profile.gate_durations:
            remapped_wires = (
                None
                if duration.wires is None
                else _remap_wires(duration.wires, local_to_region=local_to_region)
            )
            key = (duration.gate_name, remapped_wires)
            remapped_duration = GateDuration(
                gate_name=duration.gate_name,
                duration=duration.duration,
                wires=remapped_wires,
            )
            existing_duration = durations.get(key)
            if (
                existing_duration is not None
                and existing_duration.duration != remapped_duration.duration
            ):
                raise ValueError("overlapping Twin cells disagree on gate duration")
            durations[key] = remapped_duration

    expected_wires = set(range(len(region.physical_qubits)))
    if set(calibrations) != expected_wires:
        raise ValueError("Twin region composition did not cover every region qubit")
    return DeviceNoiseProfile(
        qubits=tuple(calibrations[wire] for wire in sorted(calibrations)),
        gate_durations=tuple(
            durations[key]
            for key in sorted(
                durations,
                key=lambda item: (
                    item[0],
                    () if item[1] is None else item[1],
                    item[1] is not None,
                ),
            )
        ),
        source=_region_profile_source(region.identity),
        captured_at=region.captured_at,
        time_unit=time_unit or "ns",
    )


def _readout_identity(error: ReadoutError | CorrelatedReadoutError) -> tuple[Any, ...]:
    return (type(error).__name__, error.probabilities)


def _compose_noise_model(
    cells: tuple[tuple[QPUDigitalTwin, TwinCircuitSupport], ...],
    region: TwinConnectedRegion,
    profile: DeviceNoiseProfile,
) -> NoiseModel:
    region_wire = {
        physical_qubit: wire
        for wire, physical_qubit in enumerate(region.physical_qubits)
    }
    scoped: dict[tuple[str, tuple[int, ...]], KrausChannel] = {}
    unscoped_by_gate: dict[str, dict[int, KrausChannel]] = {}
    readout: dict[tuple[int, ...], ReadoutError | CorrelatedReadoutError] = {}
    occupied_readout_wires: dict[int, tuple[int, ...]] = {}

    for cell_index, (twin, _) in enumerate(cells):
        local_to_region = {
            local: region_wire[physical]
            for local, physical in enumerate(twin.snapshot.physical_qubits)
        }
        local_unscoped: set[str] = set()
        for rule in twin.noise_model.rules:
            for raw_gate_name in rule.gate_names:
                gate_name = raw_gate_name.lower()
                if rule.wires is None:
                    if gate_name in local_unscoped:
                        raise ValueError(
                            "a Twin cell cannot repeat an unscoped gate noise rule"
                        )
                    local_unscoped.add(gate_name)
                    unscoped_by_gate.setdefault(gate_name, {})[
                        cell_index
                    ] = rule.channel
                    continue
                remapped_wires = _remap_wires(
                    rule.wires, local_to_region=local_to_region
                )
                key = (gate_name, remapped_wires)
                existing = scoped.get(key)
                if existing is not None and existing.identity != rule.channel.identity:
                    raise ValueError(
                        "overlapping Twin cells disagree on a scoped noise channel"
                    )
                scoped[key] = rule.channel

        for readout_rule in twin.noise_model.readout_rules:
            remapped_wires = _remap_wires(
                readout_rule.wires, local_to_region=local_to_region
            )
            existing_readout = readout.get(remapped_wires)
            if existing_readout is not None:
                if _readout_identity(existing_readout) != _readout_identity(
                    readout_rule.error
                ):
                    raise ValueError("overlapping Twin cells disagree on readout noise")
                continue
            for wire in remapped_wires:
                if wire in occupied_readout_wires:
                    raise ValueError(
                        "overlapping Twin readout rules cannot be composed safely"
                    )
            readout[remapped_wires] = readout_rule.error
            for wire in remapped_wires:
                occupied_readout_wires[wire] = remapped_wires

    model = NoiseModel(device_profile=profile)
    for gate_name in sorted(unscoped_by_gate):
        channels = unscoped_by_gate[gate_name]
        if set(channels) != set(range(len(cells))):
            raise ValueError(
                "unscoped gate noise must be identical in every Twin region cell"
            )
        identities = {channel.identity for channel in channels.values()}
        if len(identities) != 1:
            raise ValueError(
                "Twin region cells disagree on an unscoped gate noise channel"
            )
        model.add(gate_name, channels[0])
    for (gate_name, wires), channel in sorted(scoped.items()):
        model.add(gate_name, channel, wires=wires)
    for wires in sorted(readout):
        error = readout[wires]
        if isinstance(error, CorrelatedReadoutError):
            model.add_correlated_readout(wires, error)
        else:
            model.add_readout(wires, error)
    return model


@dataclass(frozen=True)
class TwinRegionModel:
    """A composed offline Twin model for one connected physical region.

    The model predicts measurement distributions. It carries structural scope,
    but no region-level hardware accuracy or confidence claim.
    """

    region: TwinConnectedRegion
    twin: QPUDigitalTwin
    schema: str = _REGION_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REGION_MODEL_SCHEMA:
            raise ValueError("unsupported Twin region-model schema")
        if self.twin.snapshot.provider != self.region.provider or (
            self.twin.snapshot.backend_name != self.region.backend_name
        ):
            raise ValueError("Twin region model target does not match its region")
        if self.twin.snapshot.captured_at != self.region.captured_at:
            raise ValueError("Twin region model capture does not match its region")
        if self.twin.snapshot.physical_qubits != self.region.physical_qubits:
            raise ValueError("Twin region model mapping does not match its region")

    @property
    def identity(self) -> str:
        """Return a deterministic identity for the region and composed model."""

        return _identity(
            {
                "schema": self.schema,
                "region_identity": self.region.identity,
                "snapshot_identity": self.twin.snapshot.identity,
            }
        )

    @property
    def target(self) -> str:
        """Return the canonical provider/backend target."""

        return self.region.target

    @property
    def physical_qubits(self) -> tuple[int, ...]:
        """Return the regional logical-wire to physical-qubit order."""

        return self.region.physical_qubits

    def predict(
        self,
        circuit: Any,
        *,
        physical_qubits: Sequence[int],
    ) -> TwinPrediction:
        """Predict a circuit that exactly uses this composed region mapping."""

        mapping = tuple(int(qubit) for qubit in physical_qubits)
        if mapping != self.region.physical_qubits:
            raise ValueError(
                "physical_qubits must exactly match the composed region wire order"
            )
        coverage = self.region.coverage_report(circuit, physical_qubits=mapping)
        if coverage.status != "covered":
            details = ", ".join(coverage.reasons) or "unknown scope mismatch"
            raise ValueError(f"circuit is outside the Twin region model: {details}")
        return self.twin.predict(circuit)

    def prepare_experiment(
        self,
        circuit: Any,
        *,
        physical_qubits: Sequence[int],
        name: str,
        shots: int,
    ) -> TwinExperiment:
        """Freeze one covered regional circuit for explicit QPU validation.

        The circuit, canonical OpenQASM 2.0 program, regional model prediction,
        target, physical mapping, and shot count are bound before submission.
        This method performs no provider I/O; callers must explicitly invoke
        :meth:`TwinExperiment.submit`.
        """

        self.predict(circuit, physical_qubits=physical_qubits)
        return TwinExperiment.prepare(
            self.twin,
            circuit,
            name=name,
            shots=shots,
        )

    def support_from_validation_series(
        self,
        series: TwinValidationSeries,
        circuit: Any,
        *,
        physical_qubits: Sequence[int],
    ) -> TwinCircuitSupport:
        """Qualify exact regional evidence from repeated bound QPU results.

        At least two distinct tasks are required so the resulting artifact
        carries both Twin-to-QPU agreement and QPU repeatability evidence. The
        returned support contains only the directed couplers exercised by this
        exact circuit; it does not claim arbitrary regional-circuit accuracy.
        """

        from .series import TwinValidationSeries

        if not isinstance(series, TwinValidationSeries):
            raise TypeError("series must be a TwinValidationSeries")
        mapping = tuple(int(qubit) for qubit in physical_qubits)
        if mapping != self.region.physical_qubits:
            raise ValueError(
                "physical_qubits must exactly match the composed region wire order"
            )
        coverage = self.region.coverage_report(circuit, physical_qubits=mapping)
        if coverage.status != "covered":
            details = ", ".join(coverage.reasons) or "unknown scope mismatch"
            raise ValueError(f"circuit is outside the Twin region model: {details}")
        if series.repetitions < 2:
            raise ValueError(
                "regional Twin support requires at least two distinct hardware tasks"
            )
        ir = ensure_circuit_ir(circuit)
        if series.snapshot_identity != self.twin.snapshot.identity:
            raise ValueError("validation series does not match the regional Twin")
        if series.physical_qubits != mapping:
            raise ValueError("validation series does not match the regional mapping")
        if series.circuit_identity != ir.content_hash:
            raise ValueError("validation series does not match the regional circuit")
        return TwinCircuitSupport(
            evidence=series.to_evidence(),
            directed_couplers=coverage.required_directed_couplers,
            maximum_circuit_depth=coverage.circuit_depth,
        )


def compose_region_twin(
    cells: Sequence[tuple[QPUDigitalTwin, TwinCircuitSupport]],
) -> TwinRegionModel:
    """Compose compatible local Twin cells into one connected offline model.

    This composes declared calibration and noise-model semantics only. It never
    infers correlated noise or combines local evidence into regional accuracy.
    """

    normalized = tuple(cells)
    region = compose_connected_region(normalized)
    profile = _compose_device_profile(normalized, region)
    noise_model = _compose_noise_model(normalized, region, profile)
    twin = from_noise_model(
        noise_model,
        target=region.target,
        qubits=region.physical_qubits,
    )
    return TwinRegionModel(region=region, twin=twin)


__all__ = ("compose_region_twin", "TwinRegionModel")
