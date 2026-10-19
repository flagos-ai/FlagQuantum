"""A third-party target declared through the device extension kind.

This is the ten-minute golden path for a provider that offers a real device. It
imports only the public extension namespace plus the Core capability vocabulary
that describes evidence, exactly as a third-party provider does. It does not
submit work, contact a service, read credentials, or claim anything about real
hardware: the numbers below are a *declaration* of a fictional four-qubit device,
and the evidence records that the provider measured them itself.

Two properties are worth copying:

* the gate set is expressed with the operator names the compiler can lower, and
  each name is checked against the operator schemas during validation, so a typo
  fails at the boundary instead of during a user's first compilation;
* the provider states what it *observed* separately from what it *declares*.
  Capacity and native gates are static facts a provider may declare; dtype, shot
  limits and memory are observations, and observing them requires observable-grade
  evidence.

Running :func:`flagquantum.ecosystem.extensions.target_sdk.check_target_description`
on this object is the whole admission step; nothing here registers a name, and
FlagQuantum never calls into the provider to obtain the description.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from flagquantum.core.target_capabilities import (
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
)
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)

# The provider's own calibration payload. Its digest is what the evidence
# references below point at, so a snapshot can be re-checked against the exact
# bytes the provider measured.
_CALIBRATION_PAYLOAD = b"acme-quantum/aurora-4q/calibration/2026-09-14T02:00Z"
_SPEC_PAYLOAD = b"acme-quantum/aurora-4q/datasheet/rev7"
_DEVICE_ID = "acme-aurora-4q"
_DEVICE_SCOPE = CapabilityScope(device_ids=(_DEVICE_ID,))


def _evidence(evidence_id: str, payload: bytes, level: EvidenceLevel) -> dict[str, Any]:
    """Build one evidence reference over the exact bytes the provider measured."""

    return {
        "evidence_id": evidence_id,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "level": level,
        "scope": _DEVICE_SCOPE,
    }


class AcmeAuroraTarget:
    """A declared four-qubit superconducting device with linear connectivity."""

    manifest = ExtensionManifest(
        name="acme_aurora",
        version="1.0.0",
        kind="device",
        capabilities=frozenset({"superconducting", "coupling_map", "declaration"}),
    )

    # Every device this provider offers, one declaration each. Nothing here is
    # host-owned: the provider identity comes from the manifest above.
    _declarations: ClassVar[tuple[dict[str, Any], ...]] = (
        {
            "target_id": "acme-aurora-4q",
            "device_id": _DEVICE_ID,
            "target_class": "acme_superconducting",
            "device_kind": "superconducting",
            "target_revision": "rev7",
            "environment_id": "acme-lab-a",
            "n_qubits": 4,
            # Static facts a provider is allowed to declare. This is a realistic
            # superconducting set: it has no `ry`, so the compiler's existing
            # native-gate pass has to decompose one.
            "native_gates": ("cx", "h", "rz", "s", "sdg", "sx", "x"),
            "coupling_edges": ((0, 1), (1, 2), (2, 3)),
            "declared_facts": {
                "measurements.results": ("counts", "expectation", "samples"),
                "artifacts.profiles": ("device_noise_profile",),
                "ancillas.policy": "compiler_allocated",
            },
            # Facts promoted to verified/observed, so an observation-graded
            # requirement can be satisfied by this snapshot.
            "observed_facts": {
                "device.count": 1,
                "memory.available_bytes": 137_438_953_472,
                "limits.maximum_shots": 100_000,
                "limits.maximum_program_operations": 20_000,
                # Core states the scalar width in native/storage/parameter/
                # accumulator and the complex state dtype in effective; a device
                # whose arithmetic is float32 and whose state is complex64 says
                # exactly that. It is also the width this repository's
                # statevector engines and the reference backend route accept, so
                # a program legalized for this target stays executable.
                "precision.native_dtype": "float32",
                "precision.effective_dtype": "complex64",
                "precision.storage_dtype": "float32",
                "precision.parameter_dtype": "float32",
                "precision.accumulator_dtype": "float32",
                "precision.software_mechanism": "none",
            },
            "evidence_refs": (
                EvidenceReference(
                    **_evidence(
                        "acme-calibration",
                        _CALIBRATION_PAYLOAD,
                        EvidenceLevel.OBSERVABLE,
                    )
                ),
                EvidenceReference(
                    **_evidence("acme-datasheet", _SPEC_PAYLOAD, EvidenceLevel.BASIC)
                ),
            ),
            "observed_source_ref": "acme-calibration",
            "declared_source_ref": "acme-datasheet",
        },
    )

    def __init__(self) -> None:
        self.active = False

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        supported = self.manifest.capabilities
        blockers = [
            f"capability {name!r} is unsupported"
            for name in sorted(request.required - supported)
        ]
        return CapabilityResponse(not blockers, supported, tuple(blockers))

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def close(self) -> None:
        self.active = False

    def devices(self) -> Sequence[Mapping[str, Any]]:
        """Return copies of the declarations, so a caller cannot mutate them."""

        return tuple(dict(device) for device in self._declarations)
