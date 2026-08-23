from __future__ import annotations

import pytest

from flagquantum.runtime.capabilities import (
    CapabilityEvidence,
    OperatorProfile,
    OperatorRequirement,
    preflight_operator_profile,
)


def _profile() -> OperatorProfile:
    return OperatorProfile(
        name="statevector_training_p0",
        representation="statevector",
        distribution="local",
        requirements=(
            OperatorRequirement(
                operator="aten::einsum",
                dtypes=("complex64",),
                backward=True,
            ),
        ),
    )


def test_preflight_accepts_verified_runtime_probe() -> None:
    profile = _profile()
    evidence = CapabilityEvidence(
        provider="torch_fl",
        device_type="flagos",
        profile_hash=profile.profile_hash,
        operator="aten::einsum",
        dtype="complex64",
        probe_source="runtime_probe",
        passed=True,
        forward=True,
        backward=True,
    )

    report = preflight_operator_profile(profile, (evidence,), device_type="flagos")

    assert report.supported
    assert report.evidence_ids == (evidence.evidence_id,)
    report.require_supported()


def test_preflight_rejects_names_environment_hints_and_unverified_claims() -> None:
    profile = _profile()
    claimed = CapabilityEvidence(
        provider="environment",
        device_type="flagos",
        profile_hash=profile.profile_hash,
        operator="aten::einsum",
        dtype="complex64",
        probe_source="device_name",
        passed=True,
        forward=True,
        backward=True,
    )

    report = preflight_operator_profile(profile, (claimed,), device_type="flagos")

    assert not report.supported
    assert report.blockers[0].reason == "no verified probe"
    with pytest.raises(RuntimeError, match="no verified probe"):
        report.require_supported()


def test_preflight_requires_backward_evidence_when_profile_requires_it() -> None:
    profile = _profile()
    forward_only = CapabilityEvidence(
        provider="hardware_ci",
        device_type="flagos",
        profile_hash=profile.profile_hash,
        operator="aten::einsum",
        dtype="complex64",
        probe_source="hardware_ci",
        passed=True,
        forward=True,
        backward=False,
    )

    report = preflight_operator_profile(profile, (forward_only,), device_type="flagos")

    assert not report.supported
    assert report.blockers[0].reason == "missing backward"


def test_preflight_rejects_evidence_from_a_different_profile_hash() -> None:
    profile = _profile()
    evidence = CapabilityEvidence(
        provider="torch_fl",
        device_type="flagos",
        profile_hash="0" * 64,
        operator="aten::einsum",
        dtype="complex64",
        probe_source="runtime_probe",
        passed=True,
        forward=True,
        backward=True,
    )

    report = preflight_operator_profile(profile, (evidence,), device_type="flagos")

    assert not report.supported
    assert report.evidence_ids == ()
