from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.source_preflight import (
    COMMUNITY_REVISION,
    load_source_preflight,
)

pytestmark = pytest.mark.unit

SOURCE_REVISION = "a" * 40
PLUGIN_REVISION = "b" * 40


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
