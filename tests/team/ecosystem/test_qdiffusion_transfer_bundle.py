from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.verify_transfer_bundle import (
    _write_private_json,
    verify_transfer_bundle,
)

pytestmark = pytest.mark.unit


def _archive(path: Path, *, unsafe_name: str | None = None) -> str:
    with tarfile.open(path, mode="w:gz") as archive:
        payload = b"reviewed source"
        member = tarfile.TarInfo(unsafe_name or "src/package.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, unsafe_name: str | None = None) -> Path:
    revisions = ("a" * 40, "b" * 40, "c" * 40)
    prefixes = (
        "flagquantum-qboson-",
        "kaiwu-plugin-",
        "kaiwu-community-",
    )
    entries = []
    for index, (prefix, revision) in enumerate(zip(prefixes, revisions, strict=True)):
        filename = f"{prefix}{revision[:10]}.tar.gz"
        digest = _archive(
            tmp_path / filename,
            unsafe_name=unsafe_name if index == 0 else None,
        )
        entries.append({"filename": filename, "revision": revision, "sha256": digest})
    manifest = {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": entries,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_transfer_bundle_verifies_hashes_and_safe_members(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)

    record = verify_transfer_bundle(manifest, target_host="jp-a800-171")

    assert record["safe_to_extract"] is True
    assert record["evidence_class"] == "transfer_preflight_only"
    assert record["qboson_hardware_used"] is False
    assert len(record["artifacts"]) == 3
    assert all(artifact["file_count"] == 1 for artifact in record["artifacts"])


def test_transfer_bundle_rejects_digest_mismatch(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_transfer_bundle(manifest, target_host="jp-a800-172")


@pytest.mark.parametrize("unsafe_name", ("../escape", "/absolute", "dir\\file"))
def test_transfer_bundle_rejects_unsafe_member(
    tmp_path: Path, unsafe_name: str
) -> None:
    manifest = _bundle(tmp_path, unsafe_name=unsafe_name)

    with pytest.raises(ValueError, match="archive member"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


def test_transfer_verification_record_is_exclusive_and_private(tmp_path: Path) -> None:
    output = tmp_path / "evidence" / "transfer.json"

    _write_private_json(output, {"safe_to_extract": True})

    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(output, {"safe_to_extract": True})
