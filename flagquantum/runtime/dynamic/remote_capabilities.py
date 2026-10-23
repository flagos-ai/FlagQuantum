"""Internal bridge between dynamic semantics and Target Capabilities v1.

The bridge stores its feature vector in a registered Core extension while
leaving target identity, capacity, freshness, evidence references, and
canonical serialization with the existing Target Capabilities contracts. It is
not a public API and performs no provider discovery or transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from ...core.target_capabilities import (
    CapabilityBlocker,
    CapabilityFact,
    CapabilityMatchResult,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    MatchBlockerCode,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
    match_target_capabilities,
)
from ...deployment.cloud import CloudBackendProfile
from ._conditions import instruction_condition_clauses
from .circuit import DynamicCircuit
from .conformance import DynamicFeatureSet
from .deployment import _dynamic_program_ir_blockers

DYNAMIC_REMOTE_EXTENSION = "org.flagquantum.dynamic-remote-v1"
DYNAMIC_REMOTE_EXTENSION_VERSION = "1.0"

_FEATURE_NAMES = (
    "mid_circuit_measurement",
    "reset",
    "repeated_measurement",
    "classical_feed_forward",
    "supported_conditional_gates",
    "max_condition_bits",
    "condition_values",
    "supports_disjunctive_conditions",
    "returns_mid_circuit_measurements",
)
_BOOLEAN_FEATURES = frozenset(
    {
        "mid_circuit_measurement",
        "reset",
        "repeated_measurement",
        "classical_feed_forward",
        "supports_disjunctive_conditions",
        "returns_mid_circuit_measurements",
    }
)
_COLLECTION_FEATURES = frozenset({"supported_conditional_gates", "condition_values"})
_FACT_STATES = frozenset(
    {"supported", "unsupported", "unknown", "unmeasured", "not_exposed", "missing"}
)
_EVIDENCE_LEVELS = {
    "declared": EvidenceLevel.BASIC,
    "verified": EvidenceLevel.OBSERVABLE,
    "certified": EvidenceLevel.CERTIFICATION,
}
_EVIDENCE_EXPOSURES = {
    "declared": FactExposure.DECLARED,
    "verified": FactExposure.OBSERVED,
    "certified": FactExposure.OBSERVED,
}
_STATE_BLOCKERS = {
    "unknown": "dynamic_capability_fact_unknown",
    "unmeasured": "dynamic_capability_fact_unmeasured",
    "unsupported": "dynamic_capability_fact_unsupported",
    "not_exposed": "dynamic_capability_fact_not_exposed",
}


def _isoformat(value: datetime, *, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dynamic_blocker(code: str, message: str) -> CapabilityBlocker:
    return CapabilityBlocker(code=code, message=message)


def _requirements_payload(
    circuit: DynamicCircuit,
    *,
    require_mid_circuit_measurements: bool,
) -> dict[str, Any]:
    instructions = tuple(circuit.to_ir().instructions)
    measured_bits: set[int] = set()
    repeated_measurement = False
    conditional_gates: set[str] = set()
    condition_values: set[int] = set()
    maximum_condition_bits = 0
    disjunctive = False

    for instruction in instructions:
        if instruction.name == "measure":
            bit = int(instruction.metadata["classical_bit"])
            repeated_measurement = repeated_measurement or bit in measured_bits
            measured_bits.add(bit)
        clauses = instruction_condition_clauses(instruction)
        if not clauses:
            continue
        conditional_gates.add(instruction.name)
        disjunctive = disjunctive or len(clauses) > 1
        maximum_condition_bits = max(
            maximum_condition_bits,
            *(len(clause) for clause in clauses),
        )
        condition_values.update(value for clause in clauses for _, value in clause)

    return {
        "mid_circuit_measurement": any(
            instruction.name == "measure" for instruction in instructions
        ),
        "reset": any(instruction.name == "reset" for instruction in instructions),
        "repeated_measurement": repeated_measurement,
        "classical_feed_forward": bool(conditional_gates),
        "supported_conditional_gates": sorted(conditional_gates),
        "max_condition_bits": maximum_condition_bits,
        "condition_values": sorted(condition_values),
        "supports_disjunctive_conditions": disjunctive,
        "returns_mid_circuit_measurements": require_mid_circuit_measurements,
    }


def dynamic_remote_requirements(
    circuit: DynamicCircuit,
    *,
    require_mid_circuit_measurements: bool = False,
) -> RequirementSet:
    """Derive remote dynamic requirements from validated ordered Circuit IR."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("circuit must be a DynamicCircuit")
    if type(require_mid_circuit_measurements) is not bool:
        raise TypeError("require_mid_circuit_measurements must be a boolean")
    ir_blockers = _dynamic_program_ir_blockers(circuit)
    if ir_blockers:
        raise ValueError(
            "dynamic remote requirements require an exact Program IR round trip: "
            + ", ".join(ir_blockers)
        )
    required = _requirements_payload(
        circuit,
        require_mid_circuit_measurements=require_mid_circuit_measurements,
    )
    if not any(
        (
            required["mid_circuit_measurement"],
            required["reset"],
            required["classical_feed_forward"],
        )
    ):
        raise ValueError("circuit does not contain dynamic execution semantics")
    return RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="target.class",
                operator=ComparisonOperator.EQUALS,
                value="remote_qpu",
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
            ),
            CapabilityRequirement(
                name="qubits.logical_capacity",
                operator=ComparisonOperator.AT_LEAST,
                value=circuit.n_qubits,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.COMPILER,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
            ),
        ),
        extensions={
            DYNAMIC_REMOTE_EXTENSION: {
                "schema_version": DYNAMIC_REMOTE_EXTENSION_VERSION,
                "source": "ordered CircuitIR instructions",
                "required": required,
            }
        },
    )


def _feature_values(features: DynamicFeatureSet) -> dict[str, Any]:
    return {
        "mid_circuit_measurement": features.mid_circuit_measurement,
        "reset": features.reset,
        "repeated_measurement": features.supports_repeated_measurement,
        "classical_feed_forward": bool(features.supported_conditional_gates),
        "supported_conditional_gates": list(features.supported_conditional_gates),
        "max_condition_bits": features.max_condition_bits,
        "condition_values": list(features.condition_values),
        "supports_disjunctive_conditions": features.supports_disjunctive_conditions,
        "returns_mid_circuit_measurements": (features.returns_mid_circuit_measurements),
    }


def _default_fact_state(name: str, value: Any) -> str:
    if name in _BOOLEAN_FEATURES:
        return "supported" if value is True else "unsupported"
    return "supported"


def _unsupported_value(name: str) -> Any:
    if name in _BOOLEAN_FEATURES:
        return False
    if name in _COLLECTION_FEATURES:
        return []
    return 0


def dynamic_remote_snapshot(
    backend: CloudBackendProfile,
    *,
    features: DynamicFeatureSet | None,
    captured_at: datetime,
    ttl: timedelta,
    provider_version: str,
    target_revision: str,
    environment_id: str,
    evidence_id: str,
    evidence_sha256: str,
    evidence_state: str = "declared",
    fact_states: Mapping[str, str] | None = None,
) -> TargetCapabilitySnapshot:
    """Project explicit provider facts into one device-scoped Core snapshot."""

    if not isinstance(backend, CloudBackendProfile):
        raise TypeError("backend must be a CloudBackendProfile")
    if features is not None and not isinstance(features, DynamicFeatureSet):
        raise TypeError("features must be a DynamicFeatureSet or None")
    if features is not None and not backend.supports_dynamic_circuits:
        raise ValueError(
            "detailed dynamic features contradict supports_dynamic_circuits=False"
        )
    if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
        raise ValueError("dynamic capability snapshot ttl must be positive")
    for name, value in (
        ("provider_version", provider_version),
        ("target_revision", target_revision),
        ("environment_id", environment_id),
        ("evidence_id", evidence_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if evidence_state not in _EVIDENCE_LEVELS:
        raise ValueError("evidence_state must be declared, verified, or certified")

    overrides = dict(fact_states or {})
    unknown_names = sorted(set(overrides) - set(_FEATURE_NAMES))
    if unknown_names:
        raise ValueError(
            "unknown dynamic capability fact(s): " + ", ".join(unknown_names)
        )
    invalid_states = sorted(set(overrides.values()) - _FACT_STATES)
    if invalid_states:
        raise ValueError(
            "invalid dynamic capability fact state(s): " + ", ".join(invalid_states)
        )

    target_id = f"{backend.provider}:{backend.name}"
    scope = CapabilityScope(device_ids=(target_id,))
    evidence = EvidenceReference(
        evidence_id=evidence_id,
        sha256=evidence_sha256,
        level=_EVIDENCE_LEVELS[evidence_state],
        scope=scope,
    )
    source = FactSource(
        kind=f"provider_{evidence_state}",
        ref=evidence_id,
    )
    exposure = _EVIDENCE_EXPOSURES[evidence_state]
    facts = (
        CapabilityFact(
            name="target.class",
            value="remote_qpu",
            support_status=SupportStatus.VERIFIED,
            fact_exposure=exposure,
            source=source,
        ),
        CapabilityFact(
            name="qubits.logical_capacity",
            value=backend.n_qubits,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=exposure,
            source=source,
        ),
    )

    feature_values = {} if features is None else _feature_values(features)
    dynamic_facts: dict[str, dict[str, Any]] = {}
    for name in _FEATURE_NAMES:
        if name not in feature_values and name not in overrides:
            continue
        dynamic_value: Any = feature_values.get(name)
        state = overrides.get(name, _default_fact_state(name, dynamic_value))
        if state == "missing":
            continue
        if state in {"unknown", "unmeasured", "not_exposed"}:
            dynamic_value = None
        elif state == "unsupported":
            dynamic_value = _unsupported_value(name)
        elif name not in feature_values:
            raise ValueError(f"supported fact {name!r} requires a feature value")
        dynamic_facts[name] = {
            "state": state,
            "value": dynamic_value,
            "evidence_state": evidence_state,
            "source_ref": evidence_id,
        }

    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=target_id,
            target_class="remote_qpu",
            provider=backend.provider,
            provider_version=provider_version,
            target_revision=target_revision,
            environment_id=environment_id,
        ),
        scope=scope,
        captured_at=_isoformat(captured_at, field_name="captured_at"),
        valid_until=_isoformat(captured_at + ttl, field_name="valid_until"),
        facts=facts,
        evidence_refs=(evidence,),
        extensions={
            DYNAMIC_REMOTE_EXTENSION: {
                "schema_version": DYNAMIC_REMOTE_EXTENSION_VERSION,
                "feature_set": None if features is None else features.name,
                "facts": dynamic_facts,
            }
        },
    )


def _active_requirement(name: str, value: Any) -> bool:
    if name in _BOOLEAN_FEATURES:
        return value is True
    if name in _COLLECTION_FEATURES:
        return bool(value)
    if name == "max_condition_bits":
        return type(value) is int and value > 0
    return False


def _value_satisfies(name: str, available: Any, required: Any) -> bool:
    if name in _BOOLEAN_FEATURES:
        return available is True
    if name in _COLLECTION_FEATURES:
        if not isinstance(available, list) or not isinstance(required, list):
            return False
        return set(required).issubset(available)
    if name == "max_condition_bits":
        return available is None or (
            type(available) is int and available >= int(required)
        )
    return False


def _translated_core_blocker(blocker: CapabilityBlocker) -> CapabilityBlocker:
    if blocker.code == MatchBlockerCode.SNAPSHOT_STALE.value:
        return _dynamic_blocker(
            "dynamic_capability_snapshot_stale",
            "dynamic capability snapshot has reached or passed valid_until",
        )
    if blocker.code == MatchBlockerCode.TARGET_IDENTITY_MISMATCH.value:
        return _dynamic_blocker(
            "dynamic_target_identity_mismatch",
            "dynamic capability snapshot does not identify the requested target",
        )
    return blocker


def match_dynamic_remote_capabilities(
    requirements: RequirementSet,
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime | None = None,
    expected_target_identity: TargetIdentity | None = None,
) -> CapabilityMatchResult:
    """Match Core facts and the registered dynamic extension without fallback."""

    if not isinstance(requirements, RequirementSet):
        raise TypeError("requirements must be a RequirementSet")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")
    requirement_extension = requirements.to_dict()["extensions"].get(
        DYNAMIC_REMOTE_EXTENSION
    )
    if not isinstance(requirement_extension, dict):
        raise ValueError("requirements do not contain the dynamic remote extension")

    base_requirements = RequirementSet(
        requirements=requirements.requirements,
        fallback_authorizations=requirements.fallback_authorizations,
    )
    base = match_target_capabilities(
        base_requirements,
        snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
    )
    blockers = [_translated_core_blocker(item) for item in base.blockers]
    required = requirement_extension.get("required")
    if requirement_extension.get(
        "schema_version"
    ) != DYNAMIC_REMOTE_EXTENSION_VERSION or not isinstance(required, dict):
        raise ValueError("unsupported dynamic remote requirement extension")

    snapshot_extension = snapshot.to_dict()["extensions"].get(DYNAMIC_REMOTE_EXTENSION)
    facts = (
        snapshot_extension.get("facts", {})
        if isinstance(snapshot_extension, dict)
        and snapshot_extension.get("schema_version") == DYNAMIC_REMOTE_EXTENSION_VERSION
        else {}
    )
    evidence = {item.evidence_id: item for item in snapshot.evidence_refs}
    for name in _FEATURE_NAMES:
        required_value = required.get(name)
        if not _active_requirement(name, required_value):
            continue
        fact = facts.get(name) if isinstance(facts, dict) else None
        if not isinstance(fact, dict):
            blockers.append(
                _dynamic_blocker(
                    f"dynamic_capability_fact_missing:{name}",
                    f"target snapshot does not contain required dynamic fact {name}",
                )
            )
            continue
        state = fact.get("state")
        if state in _STATE_BLOCKERS:
            blockers.append(
                _dynamic_blocker(
                    f"{_STATE_BLOCKERS[state]}:{name}",
                    f"dynamic fact {name} is {state}",
                )
            )
            continue
        if state != "supported":
            blockers.append(
                _dynamic_blocker(
                    f"dynamic_capability_fact_unknown:{name}",
                    f"dynamic fact {name} has an unrecognized state",
                )
            )
            continue
        source_ref = fact.get("source_ref")
        evidence_state = fact.get("evidence_state")
        reference = evidence.get(source_ref) if isinstance(source_ref, str) else None
        if (
            evidence_state not in _EVIDENCE_LEVELS
            or reference is None
            or reference.level is not _EVIDENCE_LEVELS[evidence_state]
        ):
            blockers.append(
                _dynamic_blocker(
                    f"dynamic_capability_fact_unmeasured:{name}",
                    f"dynamic fact {name} has no matching evidence reference",
                )
            )
            continue
        if not _value_satisfies(name, fact.get("value"), required_value):
            blockers.append(
                _dynamic_blocker(
                    f"dynamic_capability_value_mismatch:{name}",
                    f"dynamic fact {name} does not satisfy the program requirement",
                )
            )

    unique = {
        (item.code, item.message, item.capability_name): item for item in blockers
    }
    ordered = tuple(
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0], item[1], item[2] or ""))
    )
    return CapabilityMatchResult(
        executable=not ordered,
        blockers=ordered,
        satisfied_preferences=base.satisfied_preferences,
        total_preferences=base.total_preferences,
    )


__all__: tuple[str, ...] = ()
