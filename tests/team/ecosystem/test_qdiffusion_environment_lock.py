from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.verify_environment_lock import (
    load_environment_lock,
    verify_environment_lock,
    verify_frozen_environment_lock,
)

pytestmark = pytest.mark.unit


def _record() -> dict[str, object]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_environment_lock",
        "version": "1.0",
        "inventory_policy": "exact",
        "python_version": "3.10.16",
        "distributions": [
            {
                "name": "numpy",
                "version": "2.2.6",
                "approved_artifact_sha256": "a" * 64,
            },
            {
                "name": "torch",
                "version": "2.7.0",
                "approved_artifact_sha256": "b" * 64,
            },
        ],
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)


def test_environment_lock_requires_private_exact_sorted_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())

    record, digest = load_environment_lock(path)

    assert record["inventory_policy"] == "exact"
    assert len(digest) == 64

    record["distributions"] = list(reversed(record["distributions"]))  # type: ignore[arg-type]
    _write(path, record)
    with pytest.raises(ValueError, match="unique and sorted"):
        load_environment_lock(path)


def test_environment_lock_rejects_placeholder_and_public_file(tmp_path: Path) -> None:
    path = tmp_path / "environment-lock.json"
    record = _record()
    record["python_version"] = "3.10.<required>"
    _write(path, record)
    with pytest.raises(ValueError, match="exact Python"):
        load_environment_lock(path)

    _write(path, _record())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        load_environment_lock(path)


def test_environment_lock_matches_exact_runtime_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock.platform.python_version",
        lambda: "3.10.16",
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock._installed_distribution_versions",
        lambda: {"numpy": "2.2.6", "torch": "2.7.0"},
    )

    _, digest = verify_environment_lock(path)

    assert len(digest) == 64

    _, frozen_digest = verify_frozen_environment_lock(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert frozen_digest == digest

    with pytest.raises(ValueError, match="differs from frozen"):
        verify_frozen_environment_lock(path, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    ("inventory", "match"),
    (
        ({"numpy": "2.2.6"}, "missing"),
        ({"numpy": "2.2.6", "torch": "2.7.0", "extra": "1"}, "unlisted"),
        ({"numpy": "2.2.6", "torch": "2.6.0"}, "versions differ"),
    ),
)
def test_environment_lock_rejects_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inventory: dict[str, str],
    match: str,
) -> None:
    path = tmp_path / "environment-lock.json"
    _write(path, _record())
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock.platform.python_version",
        lambda: "3.10.16",
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_environment_lock._installed_distribution_versions",
        lambda: inventory,
    )

    with pytest.raises(ValueError, match=match):
        verify_environment_lock(path)
