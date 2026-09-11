"""Verified deterministic text emission after target legalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

from ..errors import CompilationError
from .openqasm import emit_openqasm
from .qcis import emit_qcis
from .schedule_legalization import (
    ScheduleLegalizationError,
    schedule_circuit_dependencies,
)
from .target_legalization import TargetLegalizationResult


class TargetEmissionError(CompilationError):
    """A legalized circuit cannot be emitted without semantic loss."""


@dataclass(frozen=True)
class TargetEmissionProfile:
    """Closed Compiler-owned description of one deterministic text emitter."""

    name: str
    backend: str
    media_type: str
    emitter: Callable[[object], str]


@dataclass(frozen=True)
class TargetEmissionResult:
    """Private emission output and audit facts, not an artifact envelope."""

    profile: str
    media_type: str
    text: str
    payload_sha256: str
    source_circuit_hash: str
    target_snapshot_id: str
    target_legalization_identity: str
    schedule_identity: str
    emission_identity: str


def _emit_openqasm2(program: object) -> str:
    return emit_openqasm(
        program,
        version=2.0,
        result_wires=_allocated_result_wires(program),
    )


def _emit_openqasm3(program: object) -> str:
    return emit_openqasm(
        program,
        version=3.0,
        result_wires=_allocated_result_wires(program),
    )


def _allocated_result_wires(program: object) -> tuple[int, ...] | None:
    routing = getattr(program, "metadata", {}).get("routing")
    if not isinstance(routing, dict) or routing.get("schema") != (
        "flagquantum_directed_routing_plan_v2"
    ):
        return None
    wires = routing.get("logical_result_physical_slots")
    if not isinstance(wires, tuple) or not wires:
        raise TargetEmissionError("allocated routing lacks logical result slots")
    return wires


_PROFILES = {
    "openqasm-2.0": TargetEmissionProfile(
        name="openqasm-2.0",
        backend="qasm",
        media_type="text/x-openqasm;version=2.0;charset=utf-8",
        emitter=_emit_openqasm2,
    ),
    "openqasm-3.0": TargetEmissionProfile(
        name="openqasm-3.0",
        backend="qasm",
        media_type="text/x-openqasm;version=3.0;charset=utf-8",
        emitter=_emit_openqasm3,
    ),
    "qcis-1.0": TargetEmissionProfile(
        name="qcis-1.0",
        backend="qcis",
        media_type="text/x-qcis;version=1.0;charset=utf-8",
        emitter=emit_qcis,
    ),
}


def _validate_semantic_profile(result: TargetLegalizationResult) -> None:
    ir = result.program
    for index, instruction in enumerate(ir.instructions):
        metadata = instruction.metadata
        if (
            instruction.name in {"measure", "reset"}
            or metadata.get("is_dynamic")
            or metadata.get("is_channel")
            or "conditions" in metadata
            or "condition_clauses" in metadata
        ):
            raise TargetEmissionError(
                f"instruction {index} cannot be represented by the static text "
                "emission profile"
            )
        if instruction.matrix is not None:
            raise TargetEmissionError(
                f"instruction {index} has an arbitrary matrix unsupported by the "
                "static text emission profile"
            )

    if ir.observables:
        raise TargetEmissionError(
            "static text emission cannot preserve observable expectation semantics"
        )
    if len(ir.measurements) != 1:
        raise TargetEmissionError(
            "static text emission requires exactly one terminal samples request"
        )
    measurement = ir.measurements[0]
    output_kind = str(measurement.metadata.get("fq_output_kind", measurement.kind))
    result_wires = _allocated_result_wires(ir)
    expected_wires = tuple(range(ir.n_wires)) if result_wires is None else result_wires
    if output_kind != "samples" or measurement.wires != expected_wires:
        raise TargetEmissionError(
            "static text emission requires terminal full-register samples"
        )


def _emission_identity(
    *,
    profile: TargetEmissionProfile,
    payload_sha256: str,
    result: TargetLegalizationResult,
) -> str:
    payload = {
        "profile": profile.name,
        "media_type": profile.media_type,
        "payload_sha256": payload_sha256,
        "source_circuit_hash": result.program.content_hash,
        "target_snapshot_id": result.target_snapshot_id,
        "target_legalization_identity": result.legalization_identity,
        "schedule_identity": result.schedule.schedule_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def emit_legalized_target(
    result: TargetLegalizationResult,
    *,
    profile: str,
) -> TargetEmissionResult:
    """Emit a target-legalized static program and bind deterministic evidence."""

    if not isinstance(result, TargetLegalizationResult):
        raise TypeError("target emission requires a TargetLegalizationResult")
    normalized_profile = str(profile).strip().lower()
    try:
        selected = _PROFILES[normalized_profile]
    except KeyError as error:
        supported = ", ".join(sorted(_PROFILES))
        raise TargetEmissionError(
            f"unsupported target emission profile {profile!r}; expected {supported}"
        ) from error
    if result.backend != selected.backend:
        raise TargetEmissionError(
            f"emission profile {selected.name!r} requires legalized backend "
            f"{selected.backend!r}, got {result.backend!r}"
        )
    if _allocated_result_wires(result.program) is not None and selected.name == (
        "qcis-1.0"
    ):
        raise TargetEmissionError(
            "qcis-1.0 cannot preserve allocated logical result projection"
        )
    if result.schedule.program is not result.program:
        raise TargetEmissionError("schedule is not bound to the legalized CircuitIR")

    try:
        current_schedule = schedule_circuit_dependencies(
            result.program,
            target_snapshot_id=result.target_snapshot_id,
        )
    except ScheduleLegalizationError as error:
        raise TargetEmissionError(
            f"legalized CircuitIR no longer satisfies scheduling: {error}"
        ) from error
    if current_schedule.schedule_identity != result.schedule.schedule_identity:
        raise TargetEmissionError(
            "legalized CircuitIR no longer matches its scheduling evidence"
        )
    _validate_semantic_profile(result)
    try:
        text = selected.emitter(result.program)
    except (NotImplementedError, OverflowError, TypeError, ValueError) as error:
        raise TargetEmissionError(
            f"{selected.name} emission failed: {error}"
        ) from error
    if not isinstance(text, str) or not text:
        raise TargetEmissionError(f"{selected.name} emitter returned an empty payload")

    payload_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return TargetEmissionResult(
        profile=selected.name,
        media_type=selected.media_type,
        text=text,
        payload_sha256=payload_sha256,
        source_circuit_hash=result.program.content_hash,
        target_snapshot_id=result.target_snapshot_id,
        target_legalization_identity=result.legalization_identity,
        schedule_identity=result.schedule.schedule_identity,
        emission_identity=_emission_identity(
            profile=selected,
            payload_sha256=payload_sha256,
            result=result,
        ),
    )


__all__ = (
    "TargetEmissionError",
    "TargetEmissionProfile",
    "TargetEmissionResult",
    "emit_legalized_target",
)
