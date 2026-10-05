"""Build one isolated, reviewable QBoson A800 source-transfer bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from examples.qdiffusion_kaiwu.verify_transfer_bundle import verify_transfer_bundle

PLUGIN_REVISION = "f047bce7b1077449967bbe9e9fab5741542b48d4"
COMMUNITY_REVISION = "b648b531c034bd6ae9b7a34fed994c717967cc72"
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(
            f"Git operation failed for reviewed checkout: {root}"
        ) from None
    return completed.stdout.strip()


def _checkout_revision(root: Path, *, expected_revision: str | None) -> str:
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(f"checkout must be an existing absolute directory: {root}")
    revision = _git(root, "rev-parse", "HEAD")
    if FULL_REVISION.fullmatch(revision) is None:
        raise ValueError(f"checkout does not expose a full Git revision: {root}")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(f"checkout revision differs from its reviewed pin: {root}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"checkout must be clean before archival: {root}")
    return revision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _archive(
    checkout: Path, *, output: Path, prefix: str, revision: str
) -> dict[str, str]:
    _git(
        checkout,
        "archive",
        "--format=tar.gz",
        f"--prefix={prefix}",
        f"--output={output}",
        revision,
    )
    output.chmod(0o600)
    return {
        "filename": output.name,
        "revision": revision,
        "sha256": _sha256(output),
    }


def build_transfer_bundle(
    *,
    flagquantum_root: Path,
    plugin_root: Path,
    community_root: Path,
    output_dir: Path,
    expected_plugin_revision: str = PLUGIN_REVISION,
    expected_community_revision: str = COMMUNITY_REVISION,
) -> Path:
    """Archive three clean Git trees and return the verified manifest path."""

    roots = (flagquantum_root, plugin_root, community_root)
    if any(not root.is_absolute() for root in roots) or not output_dir.is_absolute():
        raise ValueError("all checkout and output paths must be absolute")
    if output_dir.exists():
        raise ValueError("output directory must not already exist")
    source_revision = _checkout_revision(flagquantum_root, expected_revision=None)
    plugin_revision = _checkout_revision(
        plugin_root, expected_revision=expected_plugin_revision
    )
    community_revision = _checkout_revision(
        community_root, expected_revision=expected_community_revision
    )

    output_dir.mkdir(mode=0o700, parents=True)
    artifacts = [
        _archive(
            flagquantum_root,
            output=output_dir / f"flagquantum-qboson-{source_revision[:10]}.tar.gz",
            prefix=f"FlagQuantum-{source_revision[:10]}/",
            revision=source_revision,
        ),
        _archive(
            plugin_root,
            output=output_dir / f"kaiwu-plugin-{plugin_revision[:10]}.tar.gz",
            prefix=f"kaiwu-pytorch-plugin-{plugin_revision[:10]}/",
            revision=plugin_revision,
        ),
        _archive(
            community_root,
            output=output_dir / f"kaiwu-community-{community_revision[:10]}.tar.gz",
            prefix=f"kaiwu-community-{community_revision[:10]}/",
            revision=community_revision,
        ),
    ]
    manifest = {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": artifacts,
        "runbook": "docs/guides/QBOSON_QDIFFUSION_RUNBOOK.md",
        "limitations": [
            "The bundle has not been transferred to or executed on either target host.",
            "The bundle contains source only, not the proprietary Kaiwu SDK or credentials.",
            "The bundle contains no frozen protein dataset or model artifacts.",
        ],
    }
    manifest_path = (
        output_dir
        / f"flagquantum-qboson-a800-bundle-{source_revision[:10]}.manifest.json"
    )
    _write_exclusive(manifest_path, manifest)
    for host in ("jp-a800-171", "jp-a800-172"):
        verified = verify_transfer_bundle(manifest_path, target_host=host)
        if verified.get("safe_to_extract") is not True:
            raise RuntimeError(f"built transfer bundle failed verification for {host}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagquantum-root", required=True, type=Path)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--community-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        manifest = build_transfer_bundle(
            flagquantum_root=arguments.flagquantum_root,
            plugin_root=arguments.plugin_root,
            community_root=arguments.community_root,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified transfer bundle written to {manifest.parent}")


if __name__ == "__main__":
    main()
