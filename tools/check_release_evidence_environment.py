#!/usr/bin/env python
"""Fail-closed preflight for sealing release-certified runtime evidence."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALABILITY_ROOT = ROOT / "benchmarks" / "results" / "scalability"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _is_inside(path: Path, root: Path) -> bool:
    """Whether ``path`` names a file inside ``root``.

    Both sides are resolved before the comparison, because the answer has to be
    about the file the seal would create rather than about the spelling of the
    argument: ``benchmarks/results/scalability/../scalability`` and a symlink
    into the release directory are both inside it, and a lexical prefix test
    would accept both. A path that cannot be resolved is not inside anything.
    """

    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def readiness_errors(
    *,
    commit: str,
    signing_key_present: bool,
    device_uuids: Sequence[str],
    world_size: int,
    seal_destination: Path | None = None,
    release_root: Path | None = None,
) -> tuple[str, ...]:
    """Return release-environment blockers without exposing secret material.

    The last check is about where the sealed document would land, not about what
    the release directory already holds. "Refuse to seal while the release
    directory holds promoted JSON" was a proxy for that and failed in both
    directions: it admitted a seal written straight into an empty release
    directory, and once the first campaign had been promoted it refused every
    later seal forever, because a release directory is not emptied again. The
    invariant the campaign actually needs holds in both states -- no seal is
    written into the release directory at all -- so it is stated directly, and
    the caller that owns the destination is the one that can state it.
    """

    errors: list[str] = []
    if not _COMMIT.fullmatch(commit):
        errors.append("release evidence requires a full 40-character Git commit")
    if not signing_key_present:
        errors.append("FQ_EVIDENCE_SIGNING_KEY is not configured")
    if world_size < 1:
        errors.append("world_size must be positive")
    elif len(tuple(device_uuids)) < world_size:
        errors.append(
            f"release evidence requires {world_size} GPU UUIDs; "
            f"detected {len(tuple(device_uuids))}"
        )
    if (
        seal_destination is not None
        and release_root is not None
        and _is_inside(seal_destination, release_root)
    ):
        errors.append(
            f"seal destination {seal_destination} is inside the release directory "
            f"{release_root}; seal into a candidate directory and promote only "
            "after strict audit"
        )
    return tuple(errors)


def _output(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def environment_errors(
    *, world_size: int, seal_destination: Path | None = None
) -> tuple[str, ...]:
    commit = _output(("git", "rev-parse", "HEAD"))
    devices = tuple(
        line.strip()
        for line in _output(
            ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
        ).splitlines()
        if line.strip()
    )
    return readiness_errors(
        commit=commit,
        signing_key_present=bool(os.environ.get("FQ_EVIDENCE_SIGNING_KEY")),
        device_uuids=devices,
        world_size=world_size,
        seal_destination=seal_destination,
        release_root=SCALABILITY_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--seal-destination",
        type=Path,
        default=None,
        help=(
            "the file a seal would be written to. The release directory is not "
            "a valid destination at any point in a campaign, so naming one here "
            "is how a launcher finds that out before it measures anything."
        ),
    )
    args = parser.parse_args(argv)
    errors = environment_errors(
        world_size=args.world_size, seal_destination=args.seal_destination
    )
    if errors:
        for error in errors:
            print(f"BLOCKED: {error}")
        return 2
    print("release evidence environment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
