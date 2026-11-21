# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/git_tools.py.

"""Resolve checkout roots and comparison revisions for development commands."""

import subprocess
import sys
from pathlib import Path

from tools.development import process


def get_git_root() -> Path:
    """Return the existing absolute root of the current Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        universal_newlines=True,
        check=True,
        timeout=15,
    )
    path = Path(result.stdout.strip()).absolute()
    assert path.exists()
    return path


def comparison_revision(root: Path, revision: str | None) -> str:
    """Resolve an explicit commit or the original ordered main-branch defaults."""
    candidates = [revision] if revision else ["upstream/main", "origin/main", "main"]
    for candidate in candidates:
        result = process.run(
            ["git", "cat-file", "-t", candidate],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
            log_run_to_stderr=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "commit":
            return candidate
    if revision:
        raise ValueError(f"No revision {revision!r}")
    raise ValueError("No default revision found to compare against")


def merge_base_revision(root: Path, revision: str | None) -> str:
    """Resolve the source's commit defaults and compare at their merge base."""
    selected = comparison_revision(root, revision)
    base = process.run(
        ["git", "merge-base", selected, "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=15,
        log_run_to_stderr=False,
    ).stdout.strip()
    resolved = process.output_of(["git", "rev-parse", selected], cwd=root, timeout=15)
    if resolved == base:
        print(f"Comparing against revision {selected!r}.", file=sys.stderr)
        return selected
    print(
        f"Comparing against revision {selected!r} (merge base {base}).",
        file=sys.stderr,
    )
    return str(base)
