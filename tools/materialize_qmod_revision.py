"""Materialize a hash-checked Qmod acceptance overlay in a separate directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

_TRACKER = "qmod_compatibility_tracker.json"


def _blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def materialize(original: Path, patch: Path, manifest: Path, destination: Path) -> None:
    """Verify the pinned inputs and publish a separate, completely checked corpus."""
    original, patch, destination = (
        original.resolve(),
        patch.resolve(),
        destination.resolve(),
    )
    review = json.loads(manifest.read_text())
    patch_bytes = (
        gzip.decompress(patch.read_bytes())
        if patch.suffix == ".gz"
        else patch.read_bytes()
    )
    if hashlib.sha256(patch_bytes).hexdigest() != review["patch_sha256"]:
        raise ValueError("Qmod revision patch hash mismatch")
    if destination.is_relative_to(original):
        raise ValueError(
            "Qmod revision destination must be outside the original corpus"
        )
    if destination.exists():
        raise FileExistsError(destination)
    tracker = json.loads((original / _TRACKER).read_text())
    entries = {entry["path"]: entry for entry in tracker["files"]}
    revisions = {entry["path"]: entry for entry in review["files"]}
    if (
        len(entries) != len(tracker["files"])
        or len(revisions) != len(review["files"])
        or not revisions
        or not revisions.keys() <= entries.keys()
    ):
        raise ValueError("Qmod revision manifest has duplicate or unknown paths")
    source_paths = {
        str(path.relative_to(original / "qmod"))
        for path in (original / "qmod").rglob("*.qmod")
    }
    if source_paths != entries.keys():
        raise ValueError("Qmod corpus paths disagree with the tracker")
    for relative, entry in entries.items():
        raw = (original / "qmod" / relative).read_bytes()
        if (
            entry["source_commit"] != review["pinned_classiq_commit"]
            or _blob(raw) != entry["source_blob_sha"]
        ):
            raise ValueError(f"pinned Qmod revision mismatch for {relative}")
        if relative in revisions:
            revised = revisions[relative]
            if (
                _blob(raw) != revised["original_blob"]
                or hashlib.sha256(raw).hexdigest() != revised["original_sha256"]
            ):
                raise ValueError(f"Qmod overlay source hash mismatch for {relative}")

    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".qmod-revision-"
    ) as temporary:
        result = Path(temporary) / "corpus"
        copied = result / "qmod"
        shutil.copytree(original / "qmod", copied)
        subprocess.run(["git", "apply", "-"], input=patch_bytes, cwd=copied, check=True)
        patched_paths = {
            str(path.relative_to(copied)) for path in copied.rglob("*.qmod")
        }
        if patched_paths != entries.keys():
            raise ValueError("Qmod revision changed the corpus path inventory")
        for relative, entry in entries.items():
            raw = (copied / relative).read_bytes()
            if relative in revisions:
                if (
                    hashlib.sha256(raw).hexdigest()
                    != revisions[relative]["candidate_sha256"]
                    or _blob(raw) == entry["source_blob_sha"]
                ):
                    raise ValueError(
                        f"Qmod overlay result hash mismatch for {relative}"
                    )
                entry["upstream_source_blob_sha"] = entry["source_blob_sha"]
                entry["source_blob_sha"] = _blob(raw)
            elif _blob(raw) != entry["source_blob_sha"]:
                raise ValueError(f"Qmod overlay changed an unlisted file at {relative}")
        tracker["local_revision"] = {
            "base_upstream_commit": review["pinned_classiq_commit"],
            "approved_local_patch_sha256": review["patch_sha256"],
            "paths": sorted(revisions),
        }
        (result / _TRACKER).write_text(json.dumps(tracker, indent=2) + "\n")
        result.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "patch", "manifest", "destination"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    materialize(args.original, args.patch, args.manifest, args.destination)


if __name__ == "__main__":
    main()
