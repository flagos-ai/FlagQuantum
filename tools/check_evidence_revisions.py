#!/usr/bin/env python3
"""Require every recorded evidence revision to be obtainable or accounted for.

``tools/evidence_provenance.py`` answers one question: does the source revision an
artifact records name a commit this repository contains? Only the six checked-in
validators ask it, and each one has its artifact path written into it. An artifact
no validator names is therefore never asked, which is how nineteen artifacts under
``artifacts/`` came to record revisions this repository does not contain while
thirteen of them said nothing about it.

This gate asks the question of a directory instead of a list. Every full-length
revision recorded at any depth in a JSON artifact under a declared evidence root
must either resolve against this repository or be declared in
``evidence-revision-origins.toml``, where the declaration states the origin a
reader can obtain the revision from. An undeclared revision that does not resolve
fails the gate, so recording a pin and saying nothing about it is no longer a way
to pass.

The origins live in one table rather than inside each artifact because several of
these artifacts are regenerated evidence: a checked-in contract test rebuilds the
payload from the recorded runs and asserts equality
(``build_flagos_statevector_capacity_profile(runs).to_dict() == payload``), so
inserting a provenance key inside them would change what those artifacts must
contain. An artifact whose ``source`` mapping is not rebuilt states the origin in
``source.revision_origin`` instead, which ``tools/evidence_provenance.py`` already
reads; both records answer the same question and neither is required for a
revision this repository contains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]

try:
    from tools.evidence_provenance import (
        ORIGIN_PRODUCING_HOST_HISTORY,
        ProvenanceUnavailableError,
        is_full_revision,
        revision_resolves,
    )
except ModuleNotFoundError:  # direct script execution
    from evidence_provenance import (
        ORIGIN_PRODUCING_HOST_HISTORY,
        ProvenanceUnavailableError,
        is_full_revision,
        revision_resolves,
    )

DECLARATIONS = ROOT / "evidence-revision-origins.toml"
SCHEMA = "flagquantum.evidence_revision_origins.v1"

#: Directories whose JSON artifacts are walked. A directory is listed when its
#: JSON files are evidence for a checked-in claim rather than a tool's output.
EVIDENCE_ROOTS = (ROOT / "artifacts",)

#: The revision names a commit of an external dependency recorded beside it.
#: ``repository`` states where a reader obtains that commit.
ORIGIN_EXTERNAL_DEPENDENCY = "external_dependency"
SUPPORTED_ORIGINS = (ORIGIN_PRODUCING_HOST_HISTORY, ORIGIN_EXTERNAL_DEPENDENCY)

ARTIFACT_FIELD = "artifact"
ARTIFACT_PATH_FIELD = "path"
REVISION_FIELD = "revision"
VALUE_FIELD = "value"
ORIGIN_FIELD = "origin"
REPOSITORY_FIELD = "repository"

_REPOSITORY_PATTERN = re.compile(r"\A[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")


@dataclass(frozen=True)
class RevisionOrigin:
    """A declared origin for one revision an evidence artifact records.

    Attributes:
        artifact: Repository-relative path of the artifact that records it.
        revision: Full-length hexadecimal revision the artifact records.
        origin: One of :data:`SUPPORTED_ORIGINS`.
        repository: Where to obtain the revision, for ``external_dependency``.
    """

    artifact: str
    revision: str
    origin: str
    repository: str | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def recorded_revisions(payload: Any) -> dict[str, tuple[str, ...]]:
    """Map each full-length revision in ``payload`` to the paths that record it.

    Args:
        payload: Decoded JSON value of an evidence artifact.

    Returns:
        One entry per distinct revision, holding the dotted-and-indexed JSON paths
        that carry it in document order. Paths are reported so an error can name
        every place a revision appears instead of only the first.
    """

    found: dict[str, list[str]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif is_full_revision(node):
            found.setdefault(node, []).append(path)

    walk(payload, "")
    return {revision: tuple(paths) for revision, paths in found.items()}


def _declared_origins(document: Any) -> tuple[RevisionOrigin, ...]:
    """Return the well-formed origins ``document`` declares.

    Entries that :func:`_declaration_errors` rejects are skipped, so a caller
    reports those errors before trusting the result.
    """

    if not isinstance(document, Mapping):
        return ()
    declarations: list[RevisionOrigin] = []
    for entry in _sequence(document.get(ARTIFACT_FIELD)):
        if not isinstance(entry, Mapping):
            continue
        artifact = entry.get(ARTIFACT_PATH_FIELD)
        if not isinstance(artifact, str) or not artifact:
            continue
        for revision in _sequence(entry.get(REVISION_FIELD)):
            if not isinstance(revision, Mapping):
                continue
            value = revision.get(VALUE_FIELD)
            origin = revision.get(ORIGIN_FIELD)
            if not isinstance(value, str) or not isinstance(origin, str):
                continue
            if origin not in SUPPORTED_ORIGINS:
                continue
            repository = revision.get(REPOSITORY_FIELD)
            declarations.append(
                RevisionOrigin(
                    artifact=artifact,
                    revision=value,
                    origin=origin,
                    repository=repository if isinstance(repository, str) else None,
                )
            )
    return tuple(declarations)


def _declaration_errors(document: Any) -> tuple[str, ...]:
    """Return why the declaration document is not a usable origin record."""

    if not isinstance(document, Mapping):
        return (f"{DECLARATIONS.name} must be a TOML table",)
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append(f"{DECLARATIONS.name} schema must be {SCHEMA!r}")
    entries = document.get(ARTIFACT_FIELD)
    if not isinstance(entries, list) or not entries:
        errors.append(f"{DECLARATIONS.name} must declare at least one artifact")
        return tuple(errors)

    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"{DECLARATIONS.name} {ARTIFACT_FIELD}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{label} must be a table")
            continue
        artifact = entry.get(ARTIFACT_PATH_FIELD)
        if not isinstance(artifact, str) or not artifact:
            errors.append(f"{label} requires a string path")
            continue
        revisions = entry.get(REVISION_FIELD)
        if not isinstance(revisions, list) or not revisions:
            errors.append(f"{artifact} must declare at least one revision")
            continue
        for position, revision in enumerate(revisions):
            errors.extend(
                _revision_errors(revision, f"{artifact} {REVISION_FIELD}[{position}]")
            )
            if not isinstance(revision, Mapping):
                continue
            value = revision.get(VALUE_FIELD)
            if not isinstance(value, str):
                continue
            key = (artifact, value)
            if key in seen:
                errors.append(f"{artifact} declares revision {value} more than once")
            seen.add(key)
    return tuple(errors)


def _declaration_consistency_errors(
    document: Any,
    *,
    root: Path = ROOT,
    evidence_roots: Sequence[Path] = EVIDENCE_ROOTS,
) -> tuple[str, ...]:
    """Return why a declared origin does not match the artifact it describes."""

    errors: list[str] = []
    for declaration in _declared_origins(document):
        artifact = root / declaration.artifact
        if not artifact.is_file():
            errors.append(
                f"{declaration.artifact} is declared but not a file in this repository"
            )
            continue
        if not any(
            _is_within(artifact, evidence_root) for evidence_root in evidence_roots
        ):
            roots = ", ".join(sorted(path.name for path in evidence_roots))
            errors.append(
                f"{declaration.artifact} is declared but is outside the walked "
                f"evidence roots ({roots}), so the declaration cannot take effect"
            )
            continue
        try:
            payload = _load_json(artifact)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{declaration.artifact} could not be read as JSON: {error}")
            continue
        if declaration.revision not in recorded_revisions(payload):
            errors.append(
                f"{declaration.artifact} does not record the declared revision "
                f"{declaration.revision}"
            )
    return tuple(errors)


def _artifact_errors(
    artifact: str,
    payload: Any,
    origins: Sequence[RevisionOrigin],
    *,
    root: Path = ROOT,
    resolutions: dict[str, bool] | None = None,
) -> tuple[str, ...]:
    """Return why an artifact's recorded revisions are not accounted for.

    Args:
        artifact: Repository-relative path used to match declarations.
        payload: Decoded JSON value of the artifact.
        origins: Declarations from :func:`_declared_origins`.
        root: Repository to resolve revisions against.
        resolutions: Cache of already-resolved revisions, to keep one ``git``
            probe per distinct revision across the whole walk.

    Returns:
        One diagnostic per unaccounted revision, and for a declared revision the
        repository already contains, which needs no origin.
    """

    declared = {
        origin.revision: origin for origin in origins if origin.artifact == artifact
    }
    errors: list[str] = []
    for revision, paths in recorded_revisions(payload).items():
        where = ", ".join(paths)
        if declared.pop(revision, None) is not None:
            if _resolves(revision, root, resolutions):
                errors.append(
                    f"{artifact} declares an origin for {revision} ({where}), but this "
                    "repository contains that commit, so the revision needs no origin"
                )
            continue
        if _resolves(revision, root, resolutions):
            continue
        errors.append(
            f"{artifact} records {revision} ({where}), which names no commit in this "
            f"repository; declare it in {DECLARATIONS.name} with origin "
            f"{ORIGIN_PRODUCING_HOST_HISTORY!r} or {ORIGIN_EXTERNAL_DEPENDENCY!r}"
        )
    return tuple(errors)


def evidence_errors(
    *,
    root: Path = ROOT,
    declarations: Path = DECLARATIONS,
    evidence_roots: Sequence[Path] = EVIDENCE_ROOTS,
) -> tuple[str, ...]:
    """Return every reason a walked artifact does not account for its revisions."""

    if not declarations.is_file():
        return (f"{declarations.name} is missing",)
    try:
        document = _load_toml(declarations)
    except (OSError, tomllib.TOMLDecodeError) as error:
        return (f"{declarations.name} is not readable TOML: {error}",)
    errors = list(_declaration_errors(document))
    if errors:
        return tuple(errors)
    errors.extend(
        _declaration_consistency_errors(
            document, root=root, evidence_roots=evidence_roots
        )
    )
    origins = _declared_origins(document)
    resolutions: dict[str, bool] = {}
    for evidence_root in evidence_roots:
        for path in sorted(evidence_root.rglob("*.json")):
            artifact = _relative(path, root)
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{artifact} could not be read as JSON: {error}")
                continue
            errors.extend(
                _artifact_errors(
                    artifact,
                    payload,
                    origins,
                    root=root,
                    resolutions=resolutions,
                )
            )
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, default=DECLARATIONS)
    args = parser.parse_args(argv)
    try:
        errors = evidence_errors(declarations=args.declarations)
    except ProvenanceUnavailableError as error:
        print(f"evidence revision origins could not be checked: {error}")
        return 1
    if errors:
        print("\n".join(errors))
        return 1
    print("evidence revision origins passed")
    return 0


def _is_within(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolves(revision: str, root: Path, resolutions: dict[str, bool] | None) -> bool:
    # Annotated rather than returned directly: this module is type-checked with
    # ``--follow-imports skip``, where the imported resolver is untyped.
    if resolutions is None:
        resolved: bool = revision_resolves(revision, root)
        return resolved
    if revision not in resolutions:
        resolutions[revision] = revision_resolves(revision, root)
    return resolutions[revision]


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, list):
        return list(value)
    return ()


def _revision_errors(revision: Any, label: str) -> tuple[str, ...]:
    if not isinstance(revision, Mapping):
        return (f"{label} must be a table",)
    errors: list[str] = []
    value = revision.get(VALUE_FIELD)
    if not is_full_revision(value):
        errors.append(
            f"{label} requires a full hexadecimal revision under {VALUE_FIELD!r}, "
            f"not {value!r}"
        )
    origin = revision.get(ORIGIN_FIELD)
    if origin not in SUPPORTED_ORIGINS:
        supported = ", ".join(repr(candidate) for candidate in SUPPORTED_ORIGINS)
        errors.append(
            f"{label} {ORIGIN_FIELD} {origin!r} is not supported; use {supported}"
        )
    repository = revision.get(REPOSITORY_FIELD)
    if origin == ORIGIN_EXTERNAL_DEPENDENCY:
        if (
            not isinstance(repository, str)
            or _REPOSITORY_PATTERN.match(repository) is None
        ):
            errors.append(
                f"{label} origin {ORIGIN_EXTERNAL_DEPENDENCY!r} requires a "
                f"{REPOSITORY_FIELD!r} of the form 'owner/name', not {repository!r}"
            )
    elif repository is not None:
        errors.append(
            f"{label} {REPOSITORY_FIELD} applies only to "
            f"{ORIGIN_EXTERNAL_DEPENDENCY!r} origins"
        )
    return tuple(errors)


if __name__ == "__main__":
    sys.exit(main())
