from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.build_transfer_bundle import build_transfer_bundle
from examples.qdiffusion_kaiwu.verify_transfer_bundle import verify_transfer_bundle

pytestmark = pytest.mark.unit


def test_builder_file_entrypoint_runs_outside_repository(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "build_transfer_bundle.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--flagquantum-root" in completed.stdout


def _git(path: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "FlagQuantum Test",
        "GIT_AUTHOR_EMAIL": "test@flagquantum.invalid",
        "GIT_COMMITTER_NAME": "FlagQuantum Test",
        "GIT_COMMITTER_EMAIL": "test@flagquantum.invalid",
    }
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


def _checkout(path: Path, filename: str) -> str:
    path.mkdir()
    _git(path, "init", "--quiet")
    (path / filename).write_text("reviewed source\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "--quiet", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD")


def test_builder_archives_clean_pinned_checkouts_and_self_verifies(
    tmp_path: Path,
) -> None:
    flagquantum = tmp_path / "flagquantum"
    plugin = tmp_path / "plugin"
    community = tmp_path / "community"
    _checkout(flagquantum, "flagquantum.py")
    plugin_revision = _checkout(plugin, "plugin.py")
    community_revision = _checkout(community, "community.py")
    output = tmp_path / "bundle"

    manifest_path = build_transfer_bundle(
        flagquantum_root=flagquantum,
        plugin_root=plugin,
        community_root=community,
        output_dir=output,
        expected_plugin_revision=plugin_revision,
        expected_community_revision=community_revision,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert output.stat().st_mode & 0o777 == 0o700
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert len(manifest["artifacts"]) == 3
    assert all(
        (output / artifact["filename"]).stat().st_mode & 0o777 == 0o600
        for artifact in manifest["artifacts"]
    )
    assert (
        verify_transfer_bundle(manifest_path, target_host="jp-a800-171")[
            "safe_to_extract"
        ]
        is True
    )


def test_builder_rejects_dirty_checkout_before_creating_output(
    tmp_path: Path,
) -> None:
    flagquantum = tmp_path / "flagquantum"
    plugin = tmp_path / "plugin"
    community = tmp_path / "community"
    _checkout(flagquantum, "flagquantum.py")
    plugin_revision = _checkout(plugin, "plugin.py")
    community_revision = _checkout(community, "community.py")
    (plugin / "untracked.txt").write_text("drift", encoding="utf-8")
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="clean"):
        build_transfer_bundle(
            flagquantum_root=flagquantum,
            plugin_root=plugin,
            community_root=community,
            output_dir=output,
            expected_plugin_revision=plugin_revision,
            expected_community_revision=community_revision,
        )

    assert not output.exists()
