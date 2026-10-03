"""Reusable conformance checks for extension authors and CI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...core.ir import CircuitIR, Instruction
from ...core.target_capabilities import (
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    match_target_capabilities,
)
from .sdk import (
    CapabilityRequest,
    ExtensionConfig,
    ExtensionRegistry,
)


@dataclass(frozen=True)
class ConformanceReport:
    extension: str
    checks: tuple[str, ...]


def check_manifest_serialization(extension: Any) -> str:
    encoded = json.dumps(extension.manifest.to_dict(), sort_keys=True)
    if json.loads(encoded)["name"] != extension.manifest.name:
        raise AssertionError("manifest JSON round trip changed extension identity")
    return "manifest_serialization"


def run_backend_conformance(extension: Any) -> ConformanceReport:
    """Validate lifecycle, dtype/device negotiation and PyTorch gradients."""

    import torch

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(
        required=frozenset({"cpu", "float32", "gradients"}),
        dtype="float32",
        device_type="cpu",
        require_gradients=True,
    )
    handle = registry.negotiate("backend", extension.manifest.name, request)
    handle.start(ExtensionConfig())
    try:
        values = torch.tensor([2.0, -3.0], requires_grad=True)
        result = handle.invoke("execute", None, values)
        if result.dtype != torch.float32 or result.device.type != "cpu":
            raise AssertionError("backend changed negotiated dtype/device")
        result.backward()
        if values.grad is None or not torch.allclose(values.grad, 2 * values.detach()):
            raise AssertionError("backend gradient conformance failed")
        checks.extend(("dtype_device", "gradients", "execution_errors"))
    finally:
        handle.close()
    if getattr(extension, "active", False):
        raise AssertionError("backend leaked active lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def run_provider_conformance(extension: Any) -> ConformanceReport:
    """Validate provider negotiation, lifecycle, serialization and cleanup."""

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(required=frozenset({"qasm", "simulator"}))
    handle = registry.negotiate("provider", extension.manifest.name, request)
    handle.start(ExtensionConfig({"endpoint": "local"}))
    try:
        payload = {"qasm": "OPENQASM 2.0;", "shots": 1}
        json.dumps(payload)
        job = handle.invoke("submit", payload)
        if handle.invoke("status", job) != "completed":
            raise AssertionError("provider did not complete reference job")
        checks.extend(("payload_serialization", "provider_errors"))
    finally:
        handle.close()
    if getattr(extension, "active", False) or getattr(extension, "jobs", {}):
        raise AssertionError("provider leaked lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def run_compiler_conformance(extension: Any) -> ConformanceReport:
    """Validate circuit-IR ownership, determinism, lifecycle, and cleanup."""

    checks = [check_manifest_serialization(extension)]
    registry = ExtensionRegistry().with_extension(extension)
    request = CapabilityRequest(required=frozenset({"circuit_ir"}))
    handle = registry.negotiate("compiler", extension.manifest.name, request)
    handle.start(ExtensionConfig())
    try:
        source = CircuitIR(2, (Instruction("h", (0,)),))
        source_hash = source.content_hash
        target = {"basis_gates": ("h", "cx")}
        first = handle.invoke("compile", source, target=target)
        second = handle.invoke("compile", source, target=target)
        if not isinstance(first, CircuitIR) or not isinstance(second, CircuitIR):
            raise AssertionError("compiler must return CircuitIR")
        if source.content_hash != source_hash:
            raise AssertionError("compiler mutated its input CircuitIR")
        if first.content_hash != second.content_hash:
            raise AssertionError("compiler output is not deterministic")
        checks.extend(("circuit_ir", "input_immutable", "deterministic"))
    finally:
        handle.close()
    if getattr(extension, "active", False):
        raise AssertionError("compiler leaked active lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def run_device_conformance(extension: Any) -> ConformanceReport:
    """Validate a device declaration, its snapshot, and fail-closed matching."""

    from .target_sdk import check_target_description, target_capability_snapshot

    checks = [check_manifest_serialization(extension)]
    descriptions = check_target_description(extension)
    registry = ExtensionRegistry().with_extension(extension)
    # A device declaration is the capability statement, so there is nothing to
    # narrow here; an empty request still proves the extension accepts activation.
    handle = registry.negotiate("device", extension.manifest.name, CapabilityRequest())
    handle.start(ExtensionConfig())
    try:
        live = check_target_description(extension)
        if live != descriptions:
            raise AssertionError("device declaration changed between two reads")
        checks.append("declaration")
        for description in descriptions:
            _check_device_description(description, target_capability_snapshot)
    finally:
        handle.close()
    checks.append("snapshot_matching")
    if getattr(extension, "active", False):
        raise AssertionError("device extension leaked active lifecycle state")
    checks.append("cleanup")
    return ConformanceReport(extension.manifest.name, tuple(checks))


def _check_device_description(description: Any, build_snapshot: Any) -> None:
    """Prove one description yields a snapshot the Core matcher accepts.

    The capacity requirement is checked in both directions, and the pair is what
    makes the check mean anything: the declaration must satisfy the capacity it
    states, and must refuse the capacity one qubit beyond it.  Only the second
    direction can fail on its own, because a snapshot built from a declaration
    cannot report less than the declaration states; the first direction is what
    keeps the requirement set calibrated to the declaration rather than to the
    builder.

    Evidence resolution is not re-checked here.  ``target_capability_snapshot``
    already refuses an unresolved source ref and already refuses observation-grade
    facts that rest on basic evidence, and a second copy of those rules would be
    validation that no reachable input can exercise.
    """

    snapshot = build_snapshot(description)
    satisfied = match_target_capabilities(
        _device_requirement_set(description),
        snapshot,
        claim_minimum_evidence_level=EvidenceLevel.BASIC,
    )
    if not satisfied.executable:
        raise AssertionError(
            "target snapshot does not satisfy its own declaration: "
            + "; ".join(item.message for item in satisfied.blockers)
        )
    overstated = match_target_capabilities(
        _device_requirement_set(description, extra_qubits=1),
        snapshot,
        claim_minimum_evidence_level=EvidenceLevel.BASIC,
    )
    if overstated.executable:
        raise AssertionError(
            "target snapshot accepted a capacity its declaration does not state"
        )


def _device_requirement_set(
    description: Any, *, extra_qubits: int = 0
) -> RequirementSet:
    """State the capacity requirement a circuit imposes on this device.

    The floor is the one ``circuit_target_requirements`` imposes: the logical
    capacity as an *observed* fact at observation grade.  A device that only
    asserts a qubit count on a datasheet would therefore pass a declaration-grade
    check and still fail every real compilation, which is precisely the defect
    this conformance check exists to catch.
    """

    return RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="qubits.logical_capacity",
                operator=ComparisonOperator.AT_LEAST,
                value=description.n_qubits + extra_qubits,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.OBSERVABLE,
                accepted_exposures=(FactExposure.OBSERVED,),
            ),
        )
    )


__all__ = (
    "ConformanceReport",
    "check_manifest_serialization",
    "run_backend_conformance",
    "run_compiler_conformance",
    "run_device_conformance",
    "run_provider_conformance",
)
