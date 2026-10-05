"""Shared helpers for the two-node probe contract tests."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _manifest_digest_at(revision: str, relative_path: str) -> str:
    """Digest of a committed file as of one revision.

    A probe artifact records the digest of the contract it read, and what it read
    is the contract as of the revision the probe ran at. Reading it back from
    history is what makes that claim checkable: comparing it against the file
    that happens to be checked out now would instead assert that no commit has
    touched the contract since, which is a statement about commit order rather
    than about the measurement, and every later contract edit would fail it for
    no reason. That the current contract still declares the same shape is checked
    separately, by the probe's own observation of it, at import.
    """

    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=_ROOT,
        capture_output=True,
        check=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


@pytest.fixture(name="manifest_digest_at")
def manifest_digest_at_fixture() -> Callable[[str, str], str]:
    """Resolve the digest of a contract file at a recorded revision."""

    return _manifest_digest_at
