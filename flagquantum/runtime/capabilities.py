"""Fail-closed operator capability evidence and preflight contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Iterable, Mapping

CAPABILITY_EVIDENCE_VERSION = "1.0"
_PASSED_PROBE_SOURCES = frozenset({"runtime_probe", "hardware_ci"})


@dataclass(frozen=True)
class OperatorRequirement:
    """One atomic operator requirement in a quantum workload profile."""

    operator: str
    dtypes: tuple[str, ...]
    forward: bool = True
    backward: bool = False
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not self.operator:
            raise ValueError("operator must be non-empty")
        if not self.dtypes or any(not dtype for dtype in self.dtypes):
            raise ValueError("dtypes must contain non-empty names")
        if len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError(f"operator {self.operator!r} contains duplicate dtypes")


def _operator_requirement_from_dict(payload: Mapping[str, Any]) -> OperatorRequirement:
    allowed = {"operator", "dtypes", "forward", "backward", "deterministic"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown operator requirement fields: {unknown}")
    missing = sorted({"operator", "dtypes"} - set(payload))
    if missing:
        raise ValueError(f"missing operator requirement fields: {missing}")
    for key in ("forward", "backward", "deterministic"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"operator requirement {key} must be boolean")
    raw_dtypes = payload.get("dtypes")
    if not isinstance(raw_dtypes, list):
        raise ValueError("operator requirement dtypes must be a JSON list")
    return OperatorRequirement(
        operator=str(payload["operator"]),
        dtypes=tuple(str(dtype) for dtype in raw_dtypes),
        forward=payload.get("forward", True),
        backward=payload.get("backward", False),
        deterministic=payload.get("deterministic", False),
    )


@dataclass(frozen=True)
class OperatorProfile:
    """Machine-readable requirements for one representation/workload."""

    name: str
    representation: str
    distribution: str
    requirements: tuple[OperatorRequirement, ...]
    profile_version: str = "1.0"
    schema: str = "flagquantum_operator_profile_v1"

    def __post_init__(self) -> None:
        if self.schema != "flagquantum_operator_profile_v1":
            raise ValueError(f"unsupported operator profile schema: {self.schema!r}")
        if self.profile_version != "1.0":
            raise ValueError(
                f"unsupported operator profile version: {self.profile_version!r}"
            )
        if not self.name or not self.representation or not self.distribution:
            raise ValueError("profile identity fields must be non-empty")
        if not self.requirements:
            raise ValueError("operator profile must contain requirements")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorProfile":
        allowed = {
            "schema",
            "profile_version",
            "name",
            "representation",
            "distribution",
            "requirements",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown operator profile fields: {unknown}")
        required = {"name", "representation", "distribution", "requirements"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing operator profile fields: {missing}")
        raw_requirements = payload.get("requirements")
        if not isinstance(raw_requirements, list):
            raise ValueError("operator profile requirements must be a JSON list")
        requirements: list[OperatorRequirement] = []
        for item in raw_requirements:
            if not isinstance(item, Mapping):
                raise ValueError("each operator requirement must be a JSON object")
            requirements.append(_operator_requirement_from_dict(item))
        return cls(
            schema=str(payload.get("schema", "flagquantum_operator_profile_v1")),
            profile_version=str(payload.get("profile_version", "1.0")),
            name=str(payload["name"]),
            representation=str(payload["representation"]),
            distribution=str(payload["distribution"]),
            requirements=tuple(requirements),
        )

    def to_dict(self) -> dict[str, Any]:
        requirements = []
        for item in self.requirements:
            requirement = asdict(item)
            requirement["dtypes"] = list(item.dtypes)
            requirements.append(requirement)
        return {
            "schema": self.schema,
            "profile_version": self.profile_version,
            "name": self.name,
            "representation": self.representation,
            "distribution": self.distribution,
            "requirements": requirements,
        }

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_operator_profile(name: str) -> OperatorProfile:
    """Load a packaged profile by stable name without accepting arbitrary paths."""

    valid_characters = "abcdefghijklmnopqrstuvwxyz0123456789_"
    if not name or any(character not in valid_characters for character in name):
        raise ValueError(f"invalid operator profile name: {name!r}")
    resource = resources.files("flagquantum.runtime.profiles").joinpath(f"{name}.json")
    if not resource.is_file():
        raise KeyError(f"unknown FlagQuantum operator profile: {name!r}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"operator profile {name!r} must contain a JSON object")
    profile = OperatorProfile.from_dict(payload)
    if profile.name != name:
        raise ValueError(
            f"operator profile resource {name!r} declares name {profile.name!r}"
        )
    return profile


@dataclass(frozen=True)
class CapabilityEvidence:
    """A single operator/dtype fact obtained from an executable probe."""

    provider: str
    device_type: str
    profile_hash: str
    operator: str
    dtype: str
    probe_source: str
    passed: bool
    forward: bool
    backward: bool
    deterministic: bool | None = None
    details: str = ""
    schema_version: str = CAPABILITY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.provider,
                self.device_type,
                self.profile_hash,
                self.operator,
                self.dtype,
            )
        ):
            raise ValueError("capability evidence identity fields must be non-empty")
        if self.schema_version != CAPABILITY_EVIDENCE_VERSION:
            raise ValueError(
                f"unsupported capability evidence version: {self.schema_version!r}"
            )

    @property
    def evidence_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_verified(self) -> bool:
        return self.passed and self.probe_source in _PASSED_PROBE_SOURCES


@dataclass(frozen=True)
class PreflightBlocker:
    operator: str
    dtype: str
    reason: str


@dataclass(frozen=True)
class CapabilityPreflightReport:
    profile: str
    profile_hash: str
    device_type: str
    required_dtypes: tuple[str, ...]
    supported: bool
    evidence_ids: tuple[str, ...]
    blockers: tuple[PreflightBlocker, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "profile_hash": self.profile_hash,
            "device_type": self.device_type,
            "required_dtypes": self.required_dtypes,
            "supported": self.supported,
            "evidence_ids": self.evidence_ids,
            "blockers": tuple(asdict(item) for item in self.blockers),
        }

    def require_supported(self) -> None:
        if not self.supported:
            summary = "; ".join(
                f"{item.operator}/{item.dtype}: {item.reason}" for item in self.blockers
            )
            raise RuntimeError(f"operator capability preflight failed: {summary}")


def _index_capability_evidence(
    evidence: Iterable[CapabilityEvidence],
    *,
    device_type: str,
    profile_hash: str,
) -> tuple[
    dict[tuple[str, str], list[CapabilityEvidence]],
    dict[tuple[str, str], list[CapabilityEvidence]],
]:
    verified: dict[tuple[str, str], list[CapabilityEvidence]] = {}
    failed: dict[tuple[str, str], list[CapabilityEvidence]] = {}
    for item in evidence:
        if item.device_type != device_type or item.profile_hash != profile_hash:
            continue
        key = (item.operator, item.dtype)
        if item.is_verified:
            verified.setdefault(key, []).append(item)
        elif item.probe_source in _PASSED_PROBE_SOURCES:
            failed.setdefault(key, []).append(item)
    return verified, failed


def _match_requirement(
    requirement: OperatorRequirement,
    dtype: str,
    verified: Mapping[tuple[str, str], list[CapabilityEvidence]],
    failed: Mapping[tuple[str, str], list[CapabilityEvidence]],
) -> tuple[str | None, PreflightBlocker | None]:
    candidates = verified.get((requirement.operator, dtype), [])
    if not candidates:
        failures = failed.get((requirement.operator, dtype), [])
        reason = "no verified probe"
        if failures:
            details = sorted(item.details or "probe did not pass" for item in failures)
            reason = "probe failed: " + " | ".join(details)
        return None, PreflightBlocker(requirement.operator, dtype, reason)

    matching = [
        item
        for item in candidates
        if (not requirement.forward or item.forward)
        and (not requirement.backward or item.backward)
        and (not requirement.deterministic or item.deterministic is True)
    ]
    if matching:
        return min(item.evidence_id for item in matching), None

    missing: list[str] = []
    if requirement.forward and not any(item.forward for item in candidates):
        missing.append("forward")
    if requirement.backward and not any(item.backward for item in candidates):
        missing.append("backward")
    if requirement.deterministic and not any(
        item.deterministic is True for item in candidates
    ):
        missing.append("determinism")
    reason = (
        "missing " + ", ".join(missing)
        if missing
        else "no single probe satisfies the complete requirement"
    )
    return None, PreflightBlocker(requirement.operator, dtype, reason)


def preflight_operator_profile(
    profile: OperatorProfile,
    evidence: Iterable[CapabilityEvidence],
    *,
    device_type: str,
    required_dtypes: Iterable[str] | None = None,
) -> CapabilityPreflightReport:
    """Match every atomic requirement against verified executable evidence."""

    selected_dtypes = None
    if required_dtypes is not None:
        selected_dtypes = frozenset(str(item) for item in required_dtypes)
    if selected_dtypes is not None and not selected_dtypes:
        raise ValueError("required_dtypes must not be empty")
    profile_dtypes = {
        dtype for requirement in profile.requirements for dtype in requirement.dtypes
    }
    unknown_dtypes = sorted((selected_dtypes or set()) - profile_dtypes)
    if unknown_dtypes:
        raise ValueError(
            f"profile {profile.name!r} does not define dtypes: {unknown_dtypes}"
        )
    verified, failed = _index_capability_evidence(
        evidence,
        device_type=device_type,
        profile_hash=profile.profile_hash,
    )
    blockers: list[PreflightBlocker] = []
    accepted: list[str] = []
    for requirement in profile.requirements:
        for dtype in requirement.dtypes:
            if selected_dtypes is not None and dtype not in selected_dtypes:
                continue
            evidence_id, blocker = _match_requirement(
                requirement, dtype, verified, failed
            )
            if evidence_id is not None:
                accepted.append(evidence_id)
            if blocker is not None:
                blockers.append(blocker)
    return CapabilityPreflightReport(
        profile=profile.name,
        profile_hash=profile.profile_hash,
        device_type=device_type,
        required_dtypes=tuple(sorted(selected_dtypes or ())),
        supported=not blockers,
        evidence_ids=tuple(sorted(set(accepted))),
        blockers=tuple(blockers),
    )


__all__ = (
    "CAPABILITY_EVIDENCE_VERSION",
    "CapabilityEvidence",
    "CapabilityPreflightReport",
    "OperatorProfile",
    "OperatorRequirement",
    "PreflightBlocker",
    "load_operator_profile",
    "preflight_operator_profile",
)
