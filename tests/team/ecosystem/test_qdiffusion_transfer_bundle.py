from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.verify_extracted_bundle import verify_extracted_bundle
from examples.qdiffusion_kaiwu.verify_transfer_bundle import (
    _write_private_json,
    verify_transfer_bundle,
)

pytestmark = pytest.mark.unit


def _archive(path: Path, *, root: str, unsafe_name: str | None = None) -> str:
    with tarfile.open(path, mode="w:gz") as archive:
        payload = b"reviewed source"
        member = tarfile.TarInfo(unsafe_name or f"{root}/src/package.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(
    tmp_path: Path,
    *,
    unsafe_name: str | None = None,
    first_root: str | None = None,
) -> Path:
    revisions = ("a" * 40, "b" * 40, "c" * 40)
    prefixes = (
        "flagquantum-qboson-",
        "kaiwu-plugin-",
        "kaiwu-community-",
    )
    entries = []
    for index, (prefix, revision) in enumerate(zip(prefixes, revisions, strict=True)):
        filename = f"{prefix}{revision[:10]}.tar.gz"
        roots = ("FlagQuantum-", "kaiwu-pytorch-plugin-", "kaiwu-community-")
        archive_root = (
            first_root
            if index == 0 and first_root
            else f"{roots[index]}{revision[:10]}"
        )
        digest = _archive(
            tmp_path / filename,
            root=archive_root,
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


def test_transfer_bundle_rejects_unlisted_archive(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    _archive(tmp_path / "unexpected.tar.gz", root="unexpected")

    with pytest.raises(ValueError, match="exact manifest artifact set"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


def test_transfer_bundle_rejects_revision_mismatched_internal_root(
    tmp_path: Path,
) -> None:
    manifest = _bundle(tmp_path, first_root="FlagQuantum-wrong-root")

    with pytest.raises(ValueError, match="revision-bound root"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


def test_transfer_bundle_requires_internal_root_to_be_a_directory(
    tmp_path: Path,
) -> None:
    manifest = _bundle(tmp_path, unsafe_name=f"FlagQuantum-{'a' * 10}")

    with pytest.raises(ValueError, match="root is not a directory"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


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


def test_extracted_bundle_is_bound_to_archive_content(tmp_path: Path) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    manifest = _bundle(transfer)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extracted, filter="data")

    record = verify_extracted_bundle(
        manifest,
        extraction_root=extracted,
        target_host="jp-a800-171",
    )

    assert record["extracted_content_verified"] is True
    assert record["acceptance_evidence"] is False
    assert len(record["artifacts"]) == 3


def test_extracted_bundle_rejects_changed_or_extra_content(tmp_path: Path) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    manifest = _bundle(transfer)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extracted, filter="data")
    changed = extracted / f"FlagQuantum-{'a' * 10}" / "src" / "package.py"
    changed.write_text("modified", encoding="utf-8")

    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-172",
        )

    changed.write_text("reviewed source", encoding="utf-8")
    (extracted / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-172",
        )


def test_extracted_bundle_rejects_symlink(tmp_path: Path) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    manifest = _bundle(transfer)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extracted, filter="data")
    (extracted / "unexpected-link").symlink_to(
        extracted / f"FlagQuantum-{'a' * 10}" / "src" / "package.py"
    )

    with pytest.raises(ValueError, match="contains a symlink"):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-171",
        )
