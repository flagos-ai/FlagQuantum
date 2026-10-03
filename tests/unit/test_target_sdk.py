"""The device extension boundary: declaration, validation, snapshot, conformance.

A third-party provider reaches FlagQuantum by declaring what its device is. The
tests below hold that declaration to the two properties that make it usable and
safe:

* it is converted into the authoritative Core Target Capabilities v1 snapshot, so
  the evidence rules and the failure codes stay in Core instead of being restated
  here; and
* a fact the provider did not state stays absent and blocked rather than being
  inferred, so a device that overstates itself fails at the boundary instead of
  during a user's first compilation.

The last tests run the real consumer: a Core matcher with the evidence floor a
compiled circuit imposes. That floor is observation grade, which is why the qubit
count is an observed fact and not a declared one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.core.target_capabilities import (
    CAPABILITY_NAMES,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    match_target_capabilities,
)
from flagquantum.ecosystem.extensions import (
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
    run_device_conformance,
)
from flagquantum.ecosystem.extensions.conformance import _device_requirement_set
from flagquantum.ecosystem.extensions.target_sdk import (
    HOST_OWNED_DESCRIPTION_KEYS,
    TARGET_DESCRIPTION_MEMBER,
    TargetDescriptionError,
    check_target_description,
    target_capability_snapshot,
)

pytestmark = pytest.mark.unit

DEVICE_ID = "acme-q0"
OBSERVED_EVIDENCE = "provider-observation"
DECLARED_EVIDENCE = "provider-declaration"


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _references(
    *, observed_level: EvidenceLevel = EvidenceLevel.OBSERVABLE
) -> tuple[EvidenceReference, ...]:
    scope = CapabilityScope(device_ids=(DEVICE_ID,))
    return (
        EvidenceReference(
            evidence_id=OBSERVED_EVIDENCE,
            sha256=_digest("acme-bench"),
            level=observed_level,
            scope=scope,
        ),
        EvidenceReference(
            evidence_id=DECLARED_EVIDENCE,
            sha256=_digest("acme-spec"),
            level=EvidenceLevel.BASIC,
            scope=scope,
        ),
    )


def _declaration(**overrides: Any) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "target_id": "acme-4q",
        "device_id": DEVICE_ID,
        "target_class": "acme_superconducting",
        "device_kind": "superconducting",
        "target_revision": "r7",
        "environment_id": "lab-a",
        "n_qubits": 4,
        "native_gates": ("h", "cx", "rz"),
        "coupling_edges": ((0, 1), (1, 2), (2, 3)),
        "evidence_refs": _references(),
    }
    declaration.update(overrides)
    return declaration


class _DeclaredDevice:
    """The smallest object that satisfies the device extension protocol."""

    def __init__(
        self,
        declarations: Sequence[Mapping[str, Any]],
        *,
        kind: str = "device",
        name: str = "acme",
    ) -> None:
        self.manifest = ExtensionManifest(
            name=name, version="1.0.0", kind=kind, capabilities=frozenset()
        )
        self._declarations = declarations
        self.active = False

    def devices(self) -> Sequence[Mapping[str, Any]]:
        return self._declarations

    def negotiate(self, request: Any) -> CapabilityResponse:
        return CapabilityResponse(True, frozenset(), ())

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def close(self) -> None:
        self.active = False


def test_declaration_is_validated_and_normalised() -> None:
    device = _DeclaredDevice([_declaration(native_gates=("RZ", "cnot", "h"))])

    (description,) = check_target_description(device)

    # `cnot` is the operator schema's alias for `cx`, and the gate set is sorted
    # so two providers listing the same gates produce the same snapshot.
    assert description.native_gates == ("cx", "h", "rz")
    assert description.coupling_edges == ((0, 1), (1, 2), (2, 3))
    assert description.n_wires == description.n_qubits == 4
    # The provider identity is the host's, taken from the admitted manifest.
    assert (description.provider, description.provider_version) == ("acme", "1.0.0")
    assert TARGET_DESCRIPTION_MEMBER == "devices"


def test_snapshot_covers_every_capability_name_and_blocks_what_was_not_stated() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))

    snapshot = target_capability_snapshot(description)

    assert [fact.name for fact in snapshot.facts] == sorted(CAPABILITY_NAMES)
    by_name = {fact.name: fact for fact in snapshot.facts}
    assert by_name["device.kind"].fact_exposure is FactExposure.OBSERVED
    assert by_name["device.kind"].value == "superconducting"
    assert by_name["qubits.logical_capacity"].fact_exposure is FactExposure.OBSERVED
    assert by_name["qubits.logical_capacity"].value == 4
    assert by_name["target.class"].fact_exposure is FactExposure.DECLARED
    assert by_name["gates.native"].value == ("cx", "h", "rz")
    assert by_name["gates.native"].source.ref == DECLARED_EVIDENCE

    # A fact nobody stated is not guessed at.
    missing = by_name["memory.available_bytes"]
    assert missing.fact_exposure is FactExposure.NOT_EXPOSED
    assert missing.support_status is SupportStatus.UNKNOWN
    assert missing.value is None
    assert missing.source.ref == OBSERVED_EVIDENCE
    assert [blocker.code for blocker in missing.blockers] == [
        "provider_memory_available_bytes_not_exposed"
    ]
    assert missing.blockers[0].capability_name == "memory.available_bytes"


def test_snapshot_uses_the_supplied_clock_and_ttl() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))
    captured = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    snapshot = target_capability_snapshot(
        description, captured_at=captured, ttl=timedelta(minutes=30)
    )

    assert snapshot.captured_at == "2026-03-01T12:00:00Z"
    assert snapshot.valid_until == "2026-03-01T12:30:00Z"
    assert snapshot.scope.device_ids == (DEVICE_ID,)
    assert snapshot.target_identity.provider_version == "1.0.0"


def test_snapshot_is_deterministic_for_one_declaration() -> None:
    declaration = _declaration()
    captured = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    first = target_capability_snapshot(
        check_target_description(_DeclaredDevice([declaration]))[0],
        captured_at=captured,
    )
    second = target_capability_snapshot(
        check_target_description(_DeclaredDevice([declaration]))[0],
        captured_at=captured,
    )

    # The capture instant is part of a snapshot's identity on purpose: two reads
    # of an unchanged device at different times are different observations.
    assert first.captured_at == second.captured_at
    assert first.snapshot_id == second.snapshot_id
    assert first.to_dict()["facts"] == second.to_dict()["facts"]


@pytest.mark.parametrize("key", sorted(HOST_OWNED_DESCRIPTION_KEYS))
def test_declaration_may_not_restate_the_host_owned_provider_identity(key: str) -> None:
    device = _DeclaredDevice([_declaration(**{key: "somebody-else"})])

    with pytest.raises(TargetDescriptionError, match="host-owned fields"):
        check_target_description(device)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"declared_facts": {"qubits.logical_capacity": 4}}, "dedicated field"),
        ({"observed_facts": {"gates.native": ("h",)}}, "dedicated field"),
        ({"observed_facts": {"not.a.capability": 1}}, "unknown v1 capability"),
        (
            {"declared_facts": {"memory.available_bytes": 1024}},
            "may only declare static v1 facts",
        ),
        (
            {
                "observed_facts": {"device.count": 1},
                "declared_facts": {"device.count": 1},
            },
            "both observed and declared",
        ),
    ],
)
def test_declaration_refuses_facts_it_may_not_state(
    updates: dict[str, Any], message: str
) -> None:
    device = _DeclaredDevice([_declaration(**updates)])

    with pytest.raises(TargetDescriptionError, match=message):
        check_target_description(device)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"native_gates": ("h", "flurb")}, "unknown FlagQuantum gate"),
        ({"native_gates": ()}, "at least one gate"),
        ({"native_gates": ("h", "h")}, "must not repeat"),
        ({"coupling_edges": ((0, 9),)}, "outside the declared 4-qubit device"),
        ({"coupling_edges": ((2, 2),)}, "cannot couple wire 2 to itself"),
        ({"coupling_edges": ((0, "1"),)}, "wires must be integers"),
        ({"coupling_edges": ((0, 1, 2),)}, "pairs of wires"),
        ({"n_qubits": 0}, "n_qubits must be a positive integer"),
        ({"device_id": "  "}, "device_id must be a non-empty string"),
        ({"observed_facts": {"limits.maximum_shots": -1}}, "non-negative integer"),
        ({"observed_facts": {"limits.maximum_shots": 1.5}}, "non-negative integer"),
        ({"observed_facts": {"measurements.results": "counts"}}, "must be an array"),
        ({"declared_facts": {"ancillas.policy": 3}}, "must be a string"),
        (
            {"declared_facts": {"ancillas.policy": {"nested": "mapping"}}},
            "JSON-safe",
        ),
        (
            {"observed_facts": {"limits.maximum_shots": {"nested": 1}}},
            "JSON-safe",
        ),
        ({"evidence_refs": ("acme-calibration",)}, "EvidenceReference values"),
        (
            {"evidence_refs": _references() + _references()},
            "must be unique",
        ),
    ],
)
def test_declaration_refuses_unusable_values(
    updates: dict[str, Any], message: str
) -> None:
    device = _DeclaredDevice([_declaration(**updates)])

    with pytest.raises(TargetDescriptionError, match=message):
        check_target_description(device)


def test_a_declaration_needs_the_device_protocol_not_merely_the_kind() -> None:
    class _ManifestOnly:
        """A provider that labels itself a device but cannot name one."""

        manifest = ExtensionManifest(
            name="acme", version="1.0.0", kind="device", capabilities=frozenset()
        )

    with pytest.raises(TargetDescriptionError, match="device extension protocol"):
        check_target_description(_ManifestOnly())


def test_declaration_refuses_a_typo_in_a_field_name() -> None:
    declaration = _declaration()
    declaration["n_qubit"] = declaration.pop("n_qubits")

    with pytest.raises(TargetDescriptionError, match="unknown or missing fields"):
        check_target_description(_DeclaredDevice([declaration]))


def test_a_malformed_declaration_names_the_provider_that_sent_it() -> None:
    declaration = _declaration()
    declaration["n_qubit"] = declaration.pop("n_qubits")

    # The wrap is what makes the failure actionable in a conformance run over
    # several providers, so it is asserted rather than implied by the cause.
    with pytest.raises(TargetDescriptionError, match=r"^device extension 'acme'"):
        check_target_description(_DeclaredDevice([declaration]))


def test_an_unusable_value_names_the_provider_that_sent_it() -> None:
    with pytest.raises(
        TargetDescriptionError,
        match=r"^device extension 'acme' has an unusable declaration",
    ):
        check_target_description(_DeclaredDevice([_declaration(n_qubits=0)]))


def test_only_a_device_extension_declares_a_target() -> None:
    with pytest.raises(TargetDescriptionError, match="only a device extension"):
        check_target_description(
            _DeclaredDevice([_declaration()], kind="provider", name="acme_provider")
        )


def test_a_device_extension_must_declare_at_least_one_device() -> None:
    with pytest.raises(TargetDescriptionError, match="declares no device"):
        check_target_description(_DeclaredDevice([]))


def test_two_devices_may_not_share_a_device_id() -> None:
    device = _DeclaredDevice([_declaration(), _declaration(target_id="acme-4q-b")])

    with pytest.raises(TargetDescriptionError, match="repeated device_id"):
        check_target_description(device)


def test_observed_facts_require_observable_grade_evidence() -> None:
    device = _DeclaredDevice(
        [_declaration(evidence_refs=_references(observed_level=EvidenceLevel.BASIC))]
    )
    (description,) = check_target_description(device)

    with pytest.raises(TargetDescriptionError, match="observable or certification"):
        target_capability_snapshot(description)


def test_declaration_must_point_at_an_evidence_reference_it_supplies() -> None:
    device = _DeclaredDevice(
        [_declaration(observed_source_ref="no-such-evidence")]
    )
    (description,) = check_target_description(device)

    with pytest.raises(TargetDescriptionError, match="has no evidence reference"):
        target_capability_snapshot(description)


def test_snapshot_refuses_a_non_positive_ttl() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))

    with pytest.raises(TargetDescriptionError, match="positive timedelta"):
        target_capability_snapshot(description, ttl=timedelta(0))


def test_snapshot_refuses_a_capture_instant_without_a_timezone() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))

    # A naive instant would be compared against a tz-aware `valid_until` by every
    # consumer, so it is refused at the boundary instead of at the first plan.
    with pytest.raises(TargetDescriptionError, match="timezone-aware"):
        target_capability_snapshot(
            description, captured_at=datetime(2026, 3, 1, 12, 0, 0)
        )


def test_snapshot_names_the_providers_version_from_the_manifest() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))

    snapshot = target_capability_snapshot(description)

    # The provider identity is host-owned, so no declaration can substitute it.
    assert snapshot.target_identity.provider == "acme"
    assert snapshot.target_identity.provider_version == "1.0.0"


def test_returned_declaration_does_not_alias_the_provider_state() -> None:
    declaration = _declaration()
    device = _DeclaredDevice([declaration])

    first = check_target_description(device)[0]
    declaration["n_qubits"] = 9

    assert first.n_qubits == 4


def test_snapshot_satisfies_the_floor_a_compiled_circuit_imposes() -> None:
    """The consumer's own floor, not a weaker one the boundary chose for itself."""

    (description,) = check_target_description(_DeclaredDevice([_declaration()]))
    snapshot = target_capability_snapshot(description)
    requirement_set = _device_requirement_set(description)

    (requirement,) = requirement_set.requirements
    assert requirement.minimum_evidence_level is EvidenceLevel.OBSERVABLE
    assert requirement.source is RequirementSource.RUNTIME_PROTOCOL
    assert requirement.strength is RequirementStrength.MANDATORY
    assert requirement.operator is ComparisonOperator.AT_LEAST
    assert requirement.value == description.n_qubits

    admitted = match_target_capabilities(requirement_set, snapshot)
    assert admitted.executable

    # And the same device refuses the capacity one qubit beyond what it declared.
    overstated = match_target_capabilities(_device_requirement_set(description, extra_qubits=1), snapshot)
    assert not overstated.executable
    assert {blocker.code for blocker in overstated.blockers} == {"value_mismatch"}


def test_device_conformance_reports_the_checks_it_ran() -> None:
    device = _DeclaredDevice([_declaration()])

    report = run_device_conformance(device)

    assert report.extension == "acme"
    assert report.checks == (
        "manifest_serialization",
        "declaration",
        "snapshot_matching",
        "cleanup",
    )
    assert device.active is False


def test_device_conformance_closes_the_extension_when_a_check_fails() -> None:
    reads = 0
    device = _DeclaredDevice([_declaration()])
    declared = device.devices

    def devices() -> Sequence[Mapping[str, Any]]:
        nonlocal reads
        reads += 1
        declaration = dict(declared()[0])
        declaration["n_qubits"] = 4 + reads
        return (declaration,)

    device.devices = devices  # type: ignore[method-assign]

    with pytest.raises(AssertionError, match="changed between two reads"):
        run_device_conformance(device)
    assert reads == 2
    assert device.active is False


def test_device_conformance_refuses_an_undeclarable_device() -> None:
    with pytest.raises(TargetDescriptionError, match="unknown FlagQuantum gate"):
        run_device_conformance(
            _DeclaredDevice([_declaration(native_gates=("h", "flurb"))])
        )


def test_device_conformance_reports_a_device_that_never_closes() -> None:
    device = _DeclaredDevice([_declaration()])

    def close() -> None:
        pass

    device.close = close  # type: ignore[method-assign]

    with pytest.raises(AssertionError, match="leaked active lifecycle state"):
        run_device_conformance(device)


def test_conformance_refuses_a_snapshot_that_outgrows_its_declaration() -> None:
    """The capacity match is a check on the pair, so a lying builder is refused."""

    (description,) = check_target_description(_DeclaredDevice([_declaration()]))
    (bigger,) = check_target_description(
        _DeclaredDevice([_declaration(n_qubits=6, coupling_edges=((0, 5),))])
    )
    lying = target_capability_snapshot(bigger)

    with pytest.raises(AssertionError, match="does not state"):
        _check_device_description(description, lambda _: lying)


def test_conformance_refuses_a_snapshot_smaller_than_its_declaration() -> None:
    (description,) = check_target_description(_DeclaredDevice([_declaration()]))
    (smaller,) = check_target_description(
        _DeclaredDevice([_declaration(n_qubits=2, coupling_edges=((0, 1),))])
    )
    lying = target_capability_snapshot(smaller)

    with pytest.raises(AssertionError, match="does not satisfy its own declaration"):
        _check_device_description(description, lambda _: lying)


def test_target_sdk_exports_stay_out_of_the_stable_root() -> None:
    from flagquantum.ecosystem.extensions import target_sdk

    assert set(target_sdk.__all__).isdisjoint(fq.__all__)


def test_a_gate_sequence_must_not_be_a_bare_string() -> None:
    device = _DeclaredDevice([_declaration(native_gates="cx")])

    with pytest.raises(TargetDescriptionError, match="sequence of gate names"):
        check_target_description(device)
