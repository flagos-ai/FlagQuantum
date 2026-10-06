"""Shared helpers for the two-node probe contract tests."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.evidence_provenance import revision_resolves

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

    Raises:
        ProvenanceUnavailableError: When this checkout cannot answer -- a
            depth-1 clone holds neither the recorded revision nor any way to
            tell one it never fetched from one the repository never contained.
            The caller skips rather than reading that silence as a mismatch.
        LookupError: When a complete repository answers and the answer rebuts
            the record: the revision names no commit here, or the file is absent
            at it. That is a defect in the record, not an unavailable check.
    """

    if not revision_resolves(revision, _ROOT):
        raise LookupError(
            f"{revision} names no commit in this repository, so the digest the "
            f"artifact records for {relative_path} cannot be recomputed"
        )
    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=_ROOT,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LookupError(
            f"{relative_path} does not exist at {revision}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


@pytest.fixture(name="manifest_digest_at")
def manifest_digest_at_fixture() -> Callable[[str, str], str]:
    """Resolve the digest of a contract file at a recorded revision."""

    return _manifest_digest_at
