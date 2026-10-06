from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import build_transfer_bundle as build_module
from examples.qdiffusion_kaiwu import stable_source_tree as source_tree_module
from examples.qdiffusion_kaiwu import verify_extracted_bundle as extracted_module
from examples.qdiffusion_kaiwu import verify_transfer_bundle as transfer_module
from examples.qdiffusion_kaiwu.verify_extracted_bundle import verify_extracted_bundle
from examples.qdiffusion_kaiwu.verify_transfer_bundle import (
    MAX_TRANSFER_MANIFEST_BYTES,
    _write_private_json,
    verify_transfer_bundle,
)

pytestmark = pytest.mark.unit


def test_bundle_builder_retains_archive_snapshots_through_host_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / name for name in ("flagquantum", "plugin", "community")]
    for root in roots:
        root.mkdir()
    revisions = iter(("a" * 40, "b" * 40, "c" * 40))
    monkeypatch.setattr(
        build_module,
        "_checkout_revision",
        lambda *args, **kwargs: next(revisions),
    )

    def fake_git(root: Path, *arguments: str) -> str:
        del root
        output_argument = next(
            argument for argument in arguments if argument.startswith("--output=")
        )
        Path(output_argument.removeprefix("--output=")).write_bytes(b"archive")
        return ""

    monkeypatch.setattr(build_module, "_git", fake_git)
    calls = 0

    def verify_then_replace(manifest_path: Path, *, target_host: str):
        nonlocal calls
        del target_host
        calls += 1
        if calls == 1:
            archive = next(manifest_path.parent.glob("*.tar.gz"))
            archive.write_bytes(archive.read_bytes())
            archive.chmod(0o600)
        return {"safe_to_extract": True}

    monkeypatch.setattr(
        build_module, "verify_transfer_bundle", verify_then_replace
    )

    with pytest.raises(ValueError, match="changed after identity capture"):
        build_module.build_transfer_bundle(
            flagquantum_root=roots[0],
            plugin_root=roots[1],
            community_root=roots[2],
            output_dir=tmp_path / "bundle",
        )


def _archive(path: Path, *, root: str, unsafe_name: str | None = None) -> str:
    with tarfile.open(path, mode="w:gz") as archive:
        payload = b"reviewed source"
        member = tarfile.TarInfo(unsafe_name or f"{root}/src/package.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    path.chmod(0o600)
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
    path.chmod(0o600)
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


def test_transfer_bundle_requires_private_bounded_manifest(tmp_path: Path) -> None:
    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    public_manifest = _bundle(public_parent)
    with pytest.raises(ValueError, match="parent must be an existing private"):
        verify_transfer_bundle(public_manifest, target_host="jp-a800-171")

    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_manifest = _bundle(private_parent)
    private_manifest.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        verify_transfer_bundle(private_manifest, target_host="jp-a800-171")

    private_manifest.write_bytes(b" " * (MAX_TRANSFER_MANIFEST_BYTES + 1))
    private_manifest.chmod(0o600)
    with pytest.raises(ValueError, match="exceeds the bounded size"):
        verify_transfer_bundle(private_manifest, target_host="jp-a800-171")


def test_transfer_bundle_rejects_symlinked_manifest(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    target = tmp_path / "manifest-target.json"
    manifest.rename(target)
    manifest.symlink_to(target.name)

    with pytest.raises(ValueError, match="regular, non-symlink"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


def test_transfer_bundle_rejects_symlinked_archive(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    archive = tmp_path / payload["artifacts"][0]["filename"]
    moved = tmp_path / "reviewed-real.payload"
    archive.rename(moved)
    archive.symlink_to(moved.name)

    with pytest.raises(ValueError, match="regular, non-symlink"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


def test_transfer_bundle_rejects_archive_replacement_between_hash_and_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    archive = tmp_path / payload["artifacts"][0]["filename"]
    moved = tmp_path / "opened-archive.tar.gz"
    real_summary = transfer_module._safe_archive_summary

    def replace_after_scan(*args, **kwargs):
        summary = real_summary(*args, **kwargs)
        archive.rename(moved)
        archive.write_bytes(b"replacement")
        archive.chmod(0o600)
        return summary

    monkeypatch.setattr(transfer_module, "_safe_archive_summary", replace_after_scan)

    with pytest.raises(ValueError, match="changed during validation|binding changed"):
        verify_transfer_bundle(manifest, target_host="jp-a800-171")


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
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    output = parent / "transfer.json"

    _write_private_json(output, {"safe_to_extract": True})

    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(output, {"safe_to_extract": True})


@pytest.mark.parametrize("unsafe_kind", ("missing", "public", "symlink"))
def test_transfer_verification_requires_private_real_parent(
    tmp_path: Path, unsafe_kind: str
) -> None:
    private_parent = tmp_path / "private"
    if unsafe_kind == "missing":
        output = private_parent / "transfer.json"
    else:
        private_parent.mkdir(mode=0o700)
        if unsafe_kind == "public":
            private_parent.chmod(0o755)
            output = private_parent / "transfer.json"
        else:
            linked_parent = tmp_path / "linked"
            linked_parent.symlink_to(private_parent, target_is_directory=True)
            output = linked_parent / "transfer.json"

    with pytest.raises(ValueError, match="existing private, non-symlink"):
        _write_private_json(output, {"safe_to_extract": True})

    assert not output.exists()


def test_extracted_bundle_is_bound_to_archive_content(tmp_path: Path) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir(mode=0o700)
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
    transfer.mkdir(mode=0o700)
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
    transfer.mkdir(mode=0o700)
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


def test_extracted_bundle_rejects_manifest_drift_between_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir(mode=0o700)
    manifest = _bundle(transfer)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extracted, filter="data")

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.verify_extracted_bundle.read_private_bytes",
        lambda *args, **kwargs: b"{}",
    )
    with pytest.raises(ValueError, match="changed after bundle verification"):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-171",
        )


def test_extracted_bundle_rejects_archive_drift_after_bundle_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir(mode=0o700)
    manifest = _bundle(transfer)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    archive = transfer / payload["artifacts"][0]["filename"]
    moved = transfer / "bundle-verified-archive.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as tar:
            tar.extractall(extracted, filter="data")
    real_verify = extracted_module.verify_transfer_bundle

    def verify_then_replace(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        archive.rename(moved)
        archive.write_bytes(b"replacement")
        archive.chmod(0o600)
        return result

    monkeypatch.setattr(
        extracted_module, "verify_transfer_bundle", verify_then_replace
    )

    with pytest.raises(ValueError, match="changed after bundle verification"):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-171",
        )


def test_extracted_bundle_rejects_source_change_after_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir(mode=0o700)
    manifest = _bundle(transfer)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for archive_path in transfer.glob("*.tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extracted, filter="data")
    target = extracted / f"FlagQuantum-{'a' * 10}" / "src" / "package.py"
    real_hash = source_tree_module._stream_sha256
    calls = 0

    def mutate_after_first_extracted_hash(stream):
        nonlocal calls
        calls += 1
        digest = real_hash(stream)
        if calls == 1:
            target.write_text("changed after hashing", encoding="utf-8")
        return digest

    monkeypatch.setattr(
        source_tree_module, "_stream_sha256", mutate_after_first_extracted_hash
    )

    with pytest.raises(
        ValueError, match="changed during hashing|extracted tree changed"
    ):
        verify_extracted_bundle(
            manifest,
            extraction_root=extracted,
            target_host="jp-a800-171",
        )


def test_extracted_bundle_has_no_unsafe_no_follow_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.py"
    target.write_text("reviewed", encoding="utf-8")
    metadata = target.lstat()

    monkeypatch.delattr(source_tree_module.os, "O_NOFOLLOW")

    with pytest.raises(ValueError, match="cannot safely hash"):
        source_tree_module._hash_stable_file(
            target, metadata, label="extracted tree"
        )
