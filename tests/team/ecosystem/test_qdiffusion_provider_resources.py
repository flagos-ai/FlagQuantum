from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_budget,
    assess_provider_resources,
    build_provider_resource_gate,
    load_provider_resources,
    provider_resource_valid_until,
    validate_provider_resource_gate,
    validate_provider_resources,
)

pytestmark = pytest.mark.unit


def _record(*, sampling: int = 1) -> dict[str, object]:
    resources = []
    for target in ("SPQC-1", "SPQC-550", "SPQC-1000"):
        resources.extend(
            (
                {"target": target, "mode": "optimization", "available": 1, "used": 0},
                {
                    "target": target,
                    "mode": "sampling",
                    "available": sampling,
                    "used": 0,
                },
            )
        )
    return {
        "schema": "flagquantum.qboson_provider_resources",
        "version": "1.0",
        "source": "authenticated_resource_bill",
        "captured_at": "2026-10-06T00:00:00+00:00",
        "valid_until": "2026-10-07T00:00:00+00:00",
        "resources": resources,
        "claim_boundary": (
            "Account-resource observation only; it is not spend approval, project "
            "assignment, provider evidence, execution evidence, or acceptance evidence."
        ),
    }


def test_resource_snapshot_passes_only_with_both_smoke_modes() -> None:
    ready, reason = assess_provider_resources(
        validate_provider_resources(_record()),
        now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )

    assert ready is True
    assert reason == "provider_smoke_resources_available"


def test_resource_snapshot_exposes_its_validated_submission_deadline() -> None:
    assert provider_resource_valid_until(validate_provider_resources(_record())) == (
        datetime(2026, 10, 7, tzinfo=timezone.utc)
    )


def test_resource_snapshot_blocks_the_observed_zero_sampling_state() -> None:
    ready, reason = assess_provider_resources(
        validate_provider_resources(_record(sampling=0)),
        now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )

    assert ready is False
    assert reason == "sampling_resource_unavailable"


def test_resource_snapshot_requires_both_modes_on_one_target() -> None:
    record = _record(sampling=0)
    resources = record["resources"]
    assert isinstance(resources, list)
    for resource in resources:
        if resource["target"] == "SPQC-1":
            resource["available"] = int(resource["mode"] == "sampling")

    ready, reason = assess_provider_resources(
        validate_provider_resources(record),
        now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )

    assert ready is False
    assert reason == "common_target_resources_unavailable"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record.update(extra=True), "top-level schema"),
        (
            lambda record: record["resources"].append(record["resources"][0]),
            "cover every target and mode",
        ),
        (
            lambda record: record["resources"][0].update(available=True),
            "nonnegative integer",
        ),
        (
            lambda record: record.update(valid_until="2026-10-08T00:00:00+00:00"),
            "within 24 hours",
        ),
    ),
)
def test_resource_snapshot_rejects_ambiguous_or_unsafe_content(
    mutation: Callable[[dict[str, object]], object], message: str
) -> None:
    record = _record()
    mutation(record)

    with pytest.raises(ValueError, match=message):
        validate_provider_resources(record)


def test_private_loader_returns_the_exact_file_digest(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "resources.json"
    raw = json.dumps(_record(), sort_keys=True).encode()
    path.write_bytes(raw)
    path.chmod(0o600)

    record, digest = load_provider_resources(path)

    assert record == _record()
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("unsafe_kind", ("public-file", "public-parent", "symlink"))
def test_private_loader_rejects_unsafe_paths(tmp_path: Path, unsafe_kind: str) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "resources.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    path.chmod(0o600)
    argument = path
    if unsafe_kind == "public-file":
        path.chmod(0o644)
    elif unsafe_kind == "public-parent":
        private.chmod(0o755)
    else:
        argument = private / "resources-link.json"
        argument.symlink_to(path)

    with pytest.raises(ValueError, match="QBoson provider resource snapshot"):
        load_provider_resources(argument)


def test_resource_snapshot_expires_fail_closed() -> None:
    ready, reason = assess_provider_resources(
        validate_provider_resources(_record()),
        now=datetime(2026, 10, 7, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert ready is False
    assert reason == "provider_resource_snapshot_expired"


def test_resource_budget_requires_one_target_to_cover_the_declared_ceiling() -> None:
    record = _record(sampling=4)
    resources = record["resources"]
    assert isinstance(resources, list)
    for resource in resources:
        if resource["target"] == "SPQC-550" and resource["mode"] == "sampling":
            resource["available"] = 128

    ready, reason = assess_provider_budget(
        validate_provider_resources(record),
        mode="sampling",
        required_calls=128,
        now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )

    assert ready is True
    assert reason == "provider_sampling_budget_available"


def test_resource_budget_rejects_a_split_or_insufficient_balance() -> None:
    ready, reason = assess_provider_budget(
        validate_provider_resources(_record(sampling=127)),
        mode="sampling",
        required_calls=128,
        now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    )

    assert ready is False
    assert reason == "sampling_resource_budget_insufficient"


@pytest.mark.parametrize(
    ("mode", "required_calls", "message"),
    (
        ("unsupported", 1, "mode must be"),
        ("sampling", 0, "positive integer"),
        ("sampling", True, "positive integer"),
    ),
)
def test_resource_budget_rejects_invalid_requirements(
    mode: str, required_calls: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_provider_budget(
            validate_provider_resources(_record()),
            mode=mode,
            required_calls=required_calls,
            now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
        )


def test_provider_resource_gate_round_trips_as_a_closed_record() -> None:
    gate = build_provider_resource_gate(
        snapshot_sha256="a" * 64,
        checked_at=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
        mode="sampling",
        required_calls=128,
    )

    assert validate_provider_resource_gate(gate) == gate


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda gate: gate.update(extra=True), "invalid schema"),
        (lambda gate: gate.update(snapshot_sha256="invalid"), "SHA-256"),
        (lambda gate: gate.update(checked_at="2026-10-06T12:00:00"), "UTC"),
        (lambda gate: gate.update(required_calls=0), "positive integer"),
    ),
)
def test_provider_resource_gate_rejects_ambiguous_evidence(
    mutation: Callable[[dict[str, object]], object], message: str
) -> None:
    gate: dict[str, object] = build_provider_resource_gate(
        snapshot_sha256="a" * 64,
        checked_at=datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
        mode="sampling",
        required_calls=128,
    )
    mutation(gate)

    with pytest.raises(ValueError, match=message):
        validate_provider_resource_gate(gate)
