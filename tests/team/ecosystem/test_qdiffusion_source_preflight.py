from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qdiffusion_kaiwu import stable_source_tree as source_tree_module
from examples.qdiffusion_kaiwu.source_preflight import (
    COMMUNITY_REVISION,
    MAX_SOURCE_PREFLIGHT_BYTES,
    load_source_preflight,
    validate_common_transfer_manifest,
    validate_runtime_source_root,
    validate_transfer_manifest_record,
)

pytestmark = pytest.mark.unit

SOURCE_REVISION = "a" * 40
PLUGIN_REVISION = "b" * 40


def test_stable_file_snapshot_rejects_owner_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = source_tree_module.capture_regular_file(source, label="source")
    real_lstat = Path.lstat

    def owner_changed(path: Path):
        metadata = real_lstat(path)
        if path != source:
            return metadata
        fields = {
            name: getattr(metadata, name)
            for name in dir(metadata)
            if name.startswith("st_")
        }
        fields["st_uid"] = metadata.st_uid + 1
        return SimpleNamespace(**fields)

    monkeypatch.setattr(Path, "lstat", owner_changed)

    with pytest.raises(ValueError, match="changed after identity capture"):
        source_tree_module.revalidate_regular_file(snapshot, label="source")


def _record() -> dict[str, object]:
    revisions = (
        ("flagquantum-qboson-", "FlagQuantum-", SOURCE_REVISION),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", PLUGIN_REVISION),
        ("kaiwu-community-", "kaiwu-community-", COMMUNITY_REVISION),
    )
    return {
        "schema": "flagquantum.qboson_a800_extracted_bundle_verification",
        "version": "1.0",
        "evidence_class": "extraction_preflight_only",
        "verification_hostname": "reviewed-hostname",
        "verified_for_target_host": "jp-a800-171",
        "manifest_sha256": "c" * 64,
        "extracted_content_verified": True,
        "artifacts": [
            {
                "filename": f"{filename_prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "extracted_root": f"{root_prefix}{revision[:10]}",
                "file_count": 10,
                "content_set_sha256": "d" * 64,
            }
            for filename_prefix, root_prefix, revision in revisions
        ],
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)


def test_source_preflight_binds_host_and_all_source_revisions(tmp_path: Path) -> None:
    path = tmp_path / "source-preflight.json"
    _write(path, _record())

    record, digest = load_source_preflight(
        path,
        execution_host="jp-a800-171",
        source_revision=SOURCE_REVISION,
        plugin_revision=PLUGIN_REVISION,
    )

    assert record["extracted_content_verified"] is True
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda record: record.update(verified_for_target_host="jp-a800-172"), "host"),
        (lambda record: record.update(extracted_content_verified=False), "content"),
        (lambda record: record.update(acceptance_evidence=True), "overstates"),
        (lambda record: record.update(manifest_sha256="not-a-digest"), "digest"),
        (
            lambda record: record["artifacts"][0].update(revision="e" * 40),
            "revision",
        ),
        (
            lambda record: record["artifacts"][1].update(
                extracted_root="kaiwu-pytorch-plugin-wrong"
            ),
            "root",
        ),
    ),
)
def test_source_preflight_rejects_identity_or_evidence_drift(
    tmp_path: Path, mutation: object, match: str
) -> None:
    record = _record()
    mutation(record)  # type: ignore[operator]
    path = tmp_path / "source-preflight.json"
    _write(path, record)

    with pytest.raises(ValueError, match=match):
        load_source_preflight(
            path,
            execution_host="jp-a800-171",
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )


def test_source_preflight_must_be_private_and_not_a_symlink(tmp_path: Path) -> None:
    path = tmp_path / "source-preflight.json"
    _write(path, _record())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        load_source_preflight(
            path,
            execution_host="jp-a800-171",
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )

    path.chmod(0o600)
    link = tmp_path / "source-preflight-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_source_preflight(
            link,
            execution_host="jp-a800-171",
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )


def test_source_preflight_requires_private_parent_and_bounded_input(
    tmp_path: Path,
) -> None:
    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    public_path = public_parent / "source-preflight.json"
    _write(public_path, _record())

    with pytest.raises(ValueError, match="parent must be an existing private"):
        load_source_preflight(
            public_path,
            execution_host="jp-a800-171",
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )

    oversized = tmp_path / "oversized-source-preflight.json"
    oversized.write_bytes(b" " * (MAX_SOURCE_PREFLIGHT_BYTES + 1))
    oversized.chmod(0o600)
    with pytest.raises(ValueError, match="exceeds the bounded size"):
        load_source_preflight(
            oversized,
            execution_host="jp-a800-171",
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )


def test_both_hosts_must_share_one_transfer_manifest() -> None:
    primary = _record()
    replay = _record()
    replay["verified_for_target_host"] = "jp-a800-172"

    assert validate_common_transfer_manifest((primary, replay)) == "c" * 64

    replay["manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="share one transfer manifest"):
        validate_common_transfer_manifest((primary, replay))


def _transfer_manifest() -> dict[str, object]:
    revisions = (
        ("flagquantum-qboson-", SOURCE_REVISION),
        ("kaiwu-plugin-", PLUGIN_REVISION),
        ("kaiwu-community-", COMMUNITY_REVISION),
    )
    return {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": [
            {
                "filename": f"{prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "sha256": "f" * 64,
            }
            for prefix, revision in revisions
        ],
    }


def test_transfer_manifest_component_binds_all_source_revisions() -> None:
    record = _transfer_manifest()
    validate_transfer_manifest_record(
        record,
        source_revision=SOURCE_REVISION,
        plugin_revision=PLUGIN_REVISION,
    )

    record["artifacts"][1]["revision"] = "e" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_transfer_manifest_record(
            record,
            source_revision=SOURCE_REVISION,
            plugin_revision=PLUGIN_REVISION,
        )


@pytest.mark.parametrize(
    ("filename_prefix", "root_prefix", "revision"),
    (
        ("flagquantum-qboson-", "FlagQuantum-", SOURCE_REVISION),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", PLUGIN_REVISION),
    ),
)
def test_runtime_source_root_is_recomputed_against_preflight(
    tmp_path: Path, filename_prefix: str, root_prefix: str, revision: str
) -> None:
    root = tmp_path / f"{root_prefix}{revision[:10]}"
    root.mkdir()
    source = root / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    verified_files = [
        {
            "path": f"{root.name}/source.py",
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    record = _record()
    artifact = next(
        artifact
        for artifact in record["artifacts"]  # type: ignore[union-attr]
        if artifact["filename"].startswith(filename_prefix)
    )
    artifact["file_count"] = 1
    artifact["content_set_sha256"] = hashlib.sha256(
        json.dumps(
            verified_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    validate_runtime_source_root(
        record,  # type: ignore[arg-type]
        filename_prefix=filename_prefix,
        root=root,
    )

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content differs"):
        validate_runtime_source_root(
            record,  # type: ignore[arg-type]
            filename_prefix=filename_prefix,
            root=root,
        )


def test_runtime_source_root_rejects_change_during_stable_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / f"FlagQuantum-{SOURCE_REVISION[:10]}"
    root.mkdir()
    source = root / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    verified_files = [
        {
            "path": f"{root.name}/source.py",
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    record = _record()
    artifact = record["artifacts"][0]  # type: ignore[index]
    artifact["file_count"] = 1
    artifact["content_set_sha256"] = hashlib.sha256(
        json.dumps(
            verified_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    real_hash = source_tree_module._stream_sha256

    def mutate_after_hash(stream):
        digest = real_hash(stream)
        source.write_text("VALUE = 2\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(source_tree_module, "_stream_sha256", mutate_after_hash)

    with pytest.raises(ValueError, match="changed during hashing"):
        validate_runtime_source_root(
            record,  # type: ignore[arg-type]
            filename_prefix="flagquantum-qboson-",
            root=root,
        )
