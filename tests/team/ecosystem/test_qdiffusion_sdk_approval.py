from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.sdk_approval import (
    EXPECTED_IDENTITY,
    load_sdk_approval,
    validate_sdk_approval_record,
    verify_approved_kaiwu_distribution,
    verify_approved_project_assignment,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _approval() -> dict[str, object]:
    return {
        "schema": "flagquantum.qboson_kaiwu_sdk_approval",
        "version": "1.1",
        "distribution": "kaiwu",
        "sdk_version": "1.3.1",
        "wheel_filename": "kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl",
        "source_url": "https://pypi.org/pypi/kaiwu/1.3.1/json",
        "sha256": "7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455",
        "service_terms_url": (
            "https://platform.qboson.com/agreement?"
            "type=QBoson-SPQC-Platform-Users-Agreement"
        ),
        "service_terms_effective_date": "2026-07-09",
        "rights_reviewed_at": "2026-10-06T00:00:00Z",
        "approval_reference": "LEGAL-APPROVAL-1",
        "project_no": "CPQC-approved",
        "project_assignment_reviewed_at": "2026-10-06T00:00:00Z",
        "project_assignment_reference": "QBOSON-ASSIGNMENT-1",
        "organizational_use_approved": True,
        "isolated_container_use_approved": True,
        "host_staging_approved": True,
        "adapter_distribution_approved": True,
        "sdk_redistribution_policy": "no-sdk-redistribution",
    }


def _environment() -> dict[str, object]:
    return {
        "distributions": [
            {
                "name": "kaiwu",
                "version": "1.3.1",
                "approved_artifact_sha256": "7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455",
            }
        ]
    }


def _write_private(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def test_private_sdk_approval_loads_with_stable_identity(tmp_path: Path) -> None:
    path = (tmp_path / "sdk-approval.json").resolve()
    _write_private(path, _approval())

    record, digest = load_sdk_approval(path)

    assert record == _approval()
    assert len(digest) == 64
    verify_approved_kaiwu_distribution(_environment(), record)


def test_documented_records_use_reviewed_distribution_identity() -> None:
    standalone = json.loads(
        (
            _REPOSITORY_ROOT / "examples/qdiffusion_kaiwu/sdk_approval.example.json"
        ).read_text(encoding="utf-8")
    )
    acceptance = json.loads(
        (
            _REPOSITORY_ROOT
            / "examples/qdiffusion_kaiwu/acceptance_config.example.json"
        ).read_text(encoding="utf-8")
    )["kaiwu_sdk"]

    for field, expected in EXPECTED_IDENTITY.items():
        assert standalone[field] == expected
        assert acceptance[field] == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("wheel_filename", "other.whl", "reviewed 1.3.1 lane"),
        ("sha256", "b" * 64, "reviewed 1.3.1 lane"),
        ("sha256", "invalid", "SHA-256 digest"),
        ("rights_reviewed_at", "2026-10-06", "timezone-aware timestamp"),
        (
            "rights_reviewed_at",
            "2026-07-08T23:59:59Z",
            "predates the reviewed service terms",
        ),
        (
            "rights_reviewed_at",
            "2999-01-01T00:00:00Z",
            "review time is in the future",
        ),
        ("approval_reference", "<required>", "frozen value is required"),
        ("project_no", "<required>", "reviewed project identifier or null"),
        ("project_no", "", "reviewed project identifier or null"),
        (
            "project_assignment_reviewed_at",
            "2026-10-06",
            "timezone-aware timestamp",
        ),
        (
            "project_assignment_reviewed_at",
            "2999-01-01T00:00:00Z",
            "review time is in the future",
        ),
        (
            "project_assignment_reference",
            "<required>",
            "frozen value is required",
        ),
        ("organizational_use_approved", False, "explicit approval is required"),
        ("isolated_container_use_approved", False, "explicit approval is required"),
        ("host_staging_approved", False, "explicit approval is required"),
        ("adapter_distribution_approved", False, "explicit approval is required"),
    ),
)
def test_sdk_approval_fails_closed(field: str, value: object, message: str) -> None:
    record = _approval()
    record[field] = value

    assert any(message in error for error in validate_sdk_approval_record(record))


def test_sdk_approval_rejects_schema_extension() -> None:
    record = _approval()
    record["unreviewed_override"] = True

    assert any(
        "field set differs" in error for error in validate_sdk_approval_record(record)
    )


def test_runtime_project_must_match_reviewed_assignment() -> None:
    verify_approved_project_assignment("CPQC-approved", _approval())

    with pytest.raises(ValueError, match="differs from the reviewed"):
        verify_approved_project_assignment("CPQC-other", _approval())


def test_runtime_may_use_reviewed_account_default_assignment() -> None:
    approval = _approval()
    approval["project_no"] = None

    assert validate_sdk_approval_record(approval) == []
    verify_approved_project_assignment(None, approval)

    with pytest.raises(ValueError, match="differs from the reviewed"):
        verify_approved_project_assignment("CPQC-other", approval)


def test_runtime_cannot_omit_reviewed_explicit_assignment() -> None:
    with pytest.raises(ValueError, match="differs from the reviewed"):
        verify_approved_project_assignment(None, _approval())


@pytest.mark.parametrize(
    "unsafe_kind",
    ("relative", "public", "symlink", "public_parent", "symlink_parent"),
)
def test_sdk_approval_loader_rejects_unsafe_file(
    tmp_path: Path, unsafe_kind: str
) -> None:
    private = (tmp_path / "approval.json").resolve()
    _write_private(private, _approval())
    if unsafe_kind == "relative":
        candidate = Path("approval.json")
    elif unsafe_kind == "public":
        private.chmod(0o644)
        candidate = private
    elif unsafe_kind == "symlink":
        candidate = (tmp_path / "approval-link.json").resolve()
        candidate.symlink_to(private)
    elif unsafe_kind == "public_parent":
        tmp_path.chmod(0o755)
        candidate = private
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir(mode=0o700)
        real_file = real_parent / "approval.json"
        _write_private(real_file, _approval())
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        candidate = linked_parent / "approval.json"

    with pytest.raises(ValueError):
        load_sdk_approval(candidate)


def test_sdk_approval_loader_rejects_unbounded_input(tmp_path: Path) -> None:
    path = (tmp_path / "approval.json").resolve()
    path.write_bytes(b" " * (64 * 1024 + 1))
    path.chmod(0o600)

    with pytest.raises(ValueError, match="bounded size"):
        load_sdk_approval(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "exactly one Kaiwu"),
        ("duplicate", "exactly one Kaiwu"),
        ("version", "version differs"),
        ("artifact", "artifact differs"),
    ),
)
def test_environment_lock_must_match_sdk_approval(mutation: str, message: str) -> None:
    environment = _environment()
    distributions = environment["distributions"]
    assert isinstance(distributions, list)
    if mutation == "missing":
        distributions.clear()
    elif mutation == "duplicate":
        distributions.append(dict(distributions[0]))
    elif mutation == "version":
        distributions[0]["version"] = "1.4.1"
    else:
        distributions[0]["approved_artifact_sha256"] = "b" * 64

    with pytest.raises(ValueError, match=message):
        verify_approved_kaiwu_distribution(environment, _approval())
