#!/usr/bin/env python3
"""Trace a recorded evidence source revision against this repository.

Evidence artifacts state the source revision they were produced from. That
statement is traceability only if a reader can obtain the commit it names, but
the validators accepted any 40-character string: a revision that named no commit
in this repository passed exactly like the commit it claimed to be. Nothing
compared the record against history.

Resolution has three outcomes, and this module keeps them apart:

* the revision names a commit this repository contains;
* the repository does not contain that commit;
* the repository cannot answer (no git, no object database, a shallow clone).

The third outcome is never reported as the second. A caller that cannot check a
revision has not learned that the revision is bad, and must not report the pass
that "no finding" would otherwise produce.

An artifact whose pin genuinely comes from the producing host's history, which
this repository never contained, records that with ``revision_origin`` inside
``source``:

.. code-block:: json

    {
      "revision": "9ef1d0448011fb6a618f7651446eea7faefc3e94",
      "revision_origin": "producing_host_history",
      "archive_sha256": "...",
      "tree_dirty": false
    }

The revision is kept, so the record loses nothing, and the reason it cannot be
resolved here is stated instead of assumed. Artifacts that do not disclose an
origin are held to ``repository_history``: their revision must resolve.

Resolution and publication answer different questions, and a claim needs both.
A revision that resolves may still be one no ref reaches -- a commit that only a
deleted pull-request branch pointed at -- which the producing checkout holds and
a fresh clone never fetches. ``revision_is_published`` is the second question.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]

#: Where ``source.revision`` came from. Absent means ``repository_history``.
REVISION_ORIGIN_FIELD = "revision_origin"

#: The pin names a commit this repository contains, and it must resolve.
ORIGIN_REPOSITORY_HISTORY = "repository_history"

#: The pin names the revision the producing host reported, which this repository
#: is not expected to contain -- a pre-release export, a local commit, or an
#: internal fork. The archive hash carries the artifact's identity.
ORIGIN_PRODUCING_HOST_HISTORY = "producing_host_history"

SUPPORTED_REVISION_ORIGINS = (
    ORIGIN_REPOSITORY_HISTORY,
    ORIGIN_PRODUCING_HOST_HISTORY,
)

#: The sentinel ``_source_provenance`` in ``tools/validate_flagos_cuda_reference.py``
#: writes when the producing host cannot read its own ``git rev-parse HEAD``.
REVISION_UNAVAILABLE = "unavailable"

#: A recorded revision is a full commit hash; an abbreviated form cannot be
#: resolved without the object it was abbreviated from.
REVISION_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")

_GIT_TIMEOUT_SECONDS = 60


class ProvenanceUnavailableError(RuntimeError):
    """The repository could not report whether a revision exists."""


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise ProvenanceUnavailableError("git is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise ProvenanceUnavailableError(
            f"git {' '.join(arguments)} exceeded {_GIT_TIMEOUT_SECONDS}s"
        ) from error


def is_full_revision(value: Any) -> TypeGuard[str]:
    """Return whether ``value`` is shaped like a full hexadecimal revision."""

    return isinstance(value, str) and REVISION_PATTERN.match(value) is not None


def revision_resolves(revision: str, root: Path = ROOT) -> bool:
    """Return whether ``revision`` names a commit stored in ``root``.

    Args:
        revision: Full hexadecimal revision recorded by the evidence producer.
        root: Repository to resolve the revision against.

    Returns:
        True when the commit is present in the object database, and False when
        the repository is complete and does not contain it.

    Raises:
        ProvenanceUnavailableError: When the repository cannot answer, so that an
            unusable checkout is never mistaken for a missing commit.
    """

    probe = _git(root, "rev-parse", "--git-dir")
    if probe.returncode != 0:
        raise ProvenanceUnavailableError(
            f"{root} is not a usable git repository: {probe.stderr.strip()}"
        )
    if _git(root, "cat-file", "-e", f"{revision}^{{commit}}").returncode == 0:
        return True
    shallow = _git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        raise ProvenanceUnavailableError(
            f"{root} is a shallow clone, which cannot separate a revision it never "
            "fetched from one this repository never contained; fetch full history"
        )
    return False


def revision_is_published(revision: str, root: Path = ROOT) -> bool:
    """Return whether a ref of ``root`` reaches ``revision``.

    Presence is not obtainability. A commit that only a deleted branch pointed
    at stays in the object database of the checkout that made it, and a clone
    that fetches this repository's refs does not obtain it -- so a pin resolved
    against the producing checkout passes while the same pin is unresolvable for
    every reader. Reachability from a ref is the property a reader depends on,
    and it is what separates a revision on published history from an orphan.

    Args:
        revision: Full hexadecimal revision recorded by the evidence producer.
        root: Repository to resolve the revision against.

    Returns:
        True when some ref of ``root`` reaches the commit, and False when the
        refs of a complete repository do not.

    Raises:
        ProvenanceUnavailableError: When the repository cannot answer, so that an
            unusable checkout is never mistaken for an orphaned revision.
    """

    probe = _git(root, "rev-parse", "--git-dir")
    if probe.returncode != 0:
        raise ProvenanceUnavailableError(
            f"{root} is not a usable git repository: {probe.stderr.strip()}"
        )
    shallow = _git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        raise ProvenanceUnavailableError(
            f"{root} is a shallow clone, whose refs reach only the fetched tips and "
            "which therefore cannot separate an orphaned revision from one it never "
            "fetched; fetch full history"
        )
    containing = _git(
        root, "for-each-ref", "--contains", revision, "--format=%(refname)"
    )
    if containing.returncode != 0:
        raise ProvenanceUnavailableError(
            f"{root} could not list the refs reaching {revision}: "
            f"{containing.stderr.strip()}"
        )
    return bool(containing.stdout.strip())


def source_revision_errors(
    source: Any,
    *,
    label: str,
    root: Path = ROOT,
) -> tuple[str, ...]:
    """Return why a recorded source revision is not traceable evidence.

    Args:
        source: The artifact's ``source`` mapping.
        label: Evidence name used to prefix the diagnostics.
        root: Repository to resolve the revision against.

    Returns:
        One diagnostic per reason the record is not traceable, and an empty tuple
        when the revision resolves or the artifact discloses an origin that
        explains why it cannot.
    """

    if not isinstance(source, Mapping):
        return (f"{label} source identity is missing",)

    origin = source.get(REVISION_ORIGIN_FIELD, ORIGIN_REPOSITORY_HISTORY)
    if origin not in SUPPORTED_REVISION_ORIGINS:
        supported = ", ".join(repr(value) for value in SUPPORTED_REVISION_ORIGINS)
        return (
            f"{label} {REVISION_ORIGIN_FIELD} {origin!r} is not supported; use "
            f"one of {supported}",
        )

    revision = source.get("revision")
    if not is_full_revision(revision):
        return (
            f"{label} requires a full hexadecimal source revision, not "
            f"{revision!r}; {REVISION_UNAVAILABLE!r} is not a revision",
        )

    if origin == ORIGIN_PRODUCING_HOST_HISTORY:
        return ()

    try:
        resolvable = revision_resolves(revision, root)
    except ProvenanceUnavailableError as error:
        return (f"{label} source revision could not be checked: {error}",)
    if resolvable:
        return ()
    return (
        f"{label} source revision {revision} names no commit in this repository; "
        f'record {REVISION_ORIGIN_FIELD} as "{ORIGIN_PRODUCING_HOST_HISTORY}" if '
        "the pin comes from the producing host's history",
    )
