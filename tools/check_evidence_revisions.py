#!/usr/bin/env python3
"""Require every recorded evidence revision to be obtainable or accounted for.

``tools/evidence_provenance.py`` answers two questions about a source revision an
artifact records: does this repository contain the commit, and does a ref of this
repository reach it? Only the six checked-in validators ask either one, and each
has its artifact path written into it. An artifact no validator names is therefore
never asked, which is how nineteen artifacts under ``artifacts/`` came to record
revisions this repository does not contain while thirteen of them said nothing
about it.

This gate asks those questions of a directory instead of a list. Every full-length
revision recorded at any depth in a JSON artifact under a declared evidence root
must either be obtainable from this repository or be declared in
``evidence-revision-origins.toml``, where the declaration states the origin a
reader can obtain the revision from. An undeclared revision that is not obtainable
fails the gate, so recording a pin and saying nothing about it is no longer a way
to pass.

A root is listed when its JSON is evidence for a checked-in claim, and three are:
``artifacts/``, ``benchmarks/results/`` and ``docs/development/evidence/``. The
third was found by asking which tracked JSON outside the first two records a
revision at all, rather than by waiting for a claim to be made about it. Two
directories that question also names are deliberately left out, and the reason is
in the comment on :data:`EVIDENCE_ROOTS`: ``ci/`` holds a lock file its validators
read as input rather than a record of what a run produced, and
``examples/assets/manifest.json`` records the revision of a third-party checkpoint.
Listing either would have the gate demand that this repository account for a value
it never produced.

The gate answers the presence question from the checkout it runs in, and that
answer is not the same in every checkout. A revision absent here may still be an
unreferenced object this repository's remote serves by name, which is exactly the
state :data:`ORIGIN_UNREFERENCED_OBJECT` records. Because telling that state from a
commit no repository serves needs the network, and this gate is a static check, it
does not guess: it reports that the checkout does not hold the revision and leaves
the choice of origin to the declaration, which states how the origin was verified.

Obtainable means both answers are yes. Presence alone is not enough: a commit that
only a deleted branch pointed at stays in the object database of the checkout that
produced it, so a pin measured against that checkout resolves while every clone
that fetches this repository fails to obtain it. Reading presence as obtainability
would also make the verdict depend on which clone ran the gate, because an
unreachable object is retained on the producing host and absent elsewhere; asking
about reachability as well sends an unresolved pin and an orphaned one down the
same path, so a complete clone reaches one verdict either way. The reachability
question is asked only once presence is established, because a repository cannot
report the refs reaching a commit it does not hold.

A revision can be accounted for in two places, and the gate reads both. The first is
the table: a declaration names one artifact, or a directory that contains the
artifacts recording the revision, which is how a bulk evidence surface states one
origin once instead of repeating it for every file.

The second is the artifact itself. ``tools/evidence_provenance.py`` already reads
``source.revision_origin``, and the same convention applies beside any field that
records a revision: an artifact carrying ``base_commit`` may state
``base_commit_origin``, and one carrying ``environment.torch_fl_revision`` may state
``environment.torch_fl_revision_origin``. The table is not a replacement for that
disclosure, and this gate does not treat it as one. Both are read, an artifact that
states its origin is not also required to appear in the table, and an artifact whose
two records disagree fails rather than passing twice: a record that contradicts
itself is worse than one that says nothing, because it reads as disclosure.

The table remains necessary because several of these artifacts are regenerated
evidence: a checked-in contract test rebuilds the payload from the recorded runs and
asserts equality
(``build_flagos_statevector_capacity_profile(runs).to_dict() == payload``), so
inserting a provenance key inside them would change what those artifacts must
contain. An origin stated in either place accounts for the revision, and neither is
required for a revision this repository obtains.

The third question is how the artifact writes the revision down, and it is asked of
the roots that cite evidence as grounds for a claim. A shortened hexadecimal run
names a commit only if the reader can expand it, so a file citing one has to give the
full value too; otherwise the citation points at a revision that cannot be named and
so cannot be obtained. This is the same question as the first two asked one level
down: an artifact whose pin cannot be checked out and a citation whose pin cannot be
identified both leave the claim resting on a revision the reader does not have. The
gate finds the shortened runs that name exactly one revision this repository gives in
full, because that is what distinguishes a citation from the other
hexadecimal-looking text a file can carry, and it skips the null-commit placeholder,
which names no commit by construction.
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
        revision_is_published,
        revision_resolves,
    )
except ModuleNotFoundError:  # direct script execution
    from evidence_provenance import (
        ORIGIN_PRODUCING_HOST_HISTORY,
        ProvenanceUnavailableError,
        is_full_revision,
        revision_is_published,
        revision_resolves,
    )

DECLARATIONS = ROOT / "evidence-revision-origins.toml"
SCHEMA = "flagquantum.evidence_revision_origins.v1"

#: Directories whose JSON artifacts are walked. A directory is listed when its
#: JSON files are evidence for a checked-in claim rather than a tool's output or a
#: tool's input. ``docs/development/evidence/`` is listed because the Jiuding
#: hardware records there are what ``docs/guides/JIUDING.md``,
#: ``docs/reference/KNOWN_LIMITATIONS.md`` and the release notes cite as their
#: evidence. ``ci/`` is not listed: its JSON is read *by* the validators as an
#: input, so a lock file there is not a record of what a run produced.
#: ``examples/assets/`` is not listed either: that manifest records the revision of
#: a third-party checkpoint, which this repository cannot be asked to account for.
EVIDENCE_ROOTS = (
    ROOT / "artifacts",
    ROOT / "benchmarks" / "results",
    ROOT / "docs" / "development" / "evidence",
)

#: Roots whose citations of a revision must be complete. This answers a different
#: question from :data:`EVIDENCE_ROOTS`, so it is a different tuple: these are the
#: roots that cite a revision as the grounds for a checked-in claim, so a reader who
#: follows a citation from one of them has to be able to obtain what it names. A
#: shortened revision is obtainable only when the file that cites it also gives the
#: full value, because nothing else tells the reader which commit the prefix names.
CITATION_ROOTS = (
    ROOT / "artifacts",
    ROOT / "benchmarks" / "results",
    ROOT / "docs" / "development" / "evidence",
    ROOT / "docs" / "reference",
    ROOT / "docs" / "roadmap",
)

#: File types that can carry a citation of a revision.
CITATION_SUFFIXES = frozenset(
    {".json", ".md", ".markdown", ".rst", ".toml", ".txt", ".yaml", ".yml"}
)

#: Directories never descended into while looking for the revisions a repository
#: cites. A full-length revision under one of these is not a citation a reader
#: follows, and ``.git`` in particular holds an object database rather than text.
CITATION_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)

#: A full-length revision, not part of a longer hexadecimal run.
_FULL_REVISION_PATTERN = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")

#: A hexadecimal run too short to be the revision itself, given in full elsewhere.
_SHORT_REVISION_PATTERN = re.compile(
    r"(?<![0-9a-fA-F])([0-9a-fA-F]{7,39})(?![0-9a-fA-F])"
)

#: The revision names a commit that no ref of this repository reaches, so neither a
#: clone nor a plain fetch obtains it. The remote does serve the object to
#: ``git fetch origin <revision>``, which is the only route to it and one that lasts
#: only while the remote retains unreferenced objects. Whether the checkout running
#: this gate also holds the object depends on that checkout and not on the remote, so
#: the gate cannot tell this origin from a commit no repository serves; the
#: declaration is established by counted fetches of the revision from the remote by
#: name, whose method ``evidence-revision-origins.toml`` states, because a failed
#: fetch is a dropped connection or a transient error as readily as a refusal.
ORIGIN_UNREFERENCED_OBJECT = "unreferenced_object"

#: The revision names a commit of an external dependency recorded beside it.
#: ``repository`` states where a reader obtains that commit.
ORIGIN_EXTERNAL_DEPENDENCY = "external_dependency"
SUPPORTED_ORIGINS = (
    ORIGIN_PRODUCING_HOST_HISTORY,
    ORIGIN_UNREFERENCED_OBJECT,
    ORIGIN_EXTERNAL_DEPENDENCY,
)

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
        artifact: Repository-relative path of the artifact that records it, or of a
            directory that contains the artifacts recording it.
        revision: Full-length hexadecimal revision the artifact records.
        origin: One of :data:`SUPPORTED_ORIGINS`.
        repository: Where to obtain the revision, for ``external_dependency``.
    """

    artifact: str
    revision: str
    origin: str
    repository: str | None = None


@dataclass(frozen=True)
class RecordedOrigin:
    """An origin an artifact states inside itself for a revision it records.

    Attributes:
        revision: Full-length hexadecimal revision the artifact records.
        field: JSON path of the field that records the revision.
        origin: Value of the ``<field>_origin`` field beside it.
    """

    revision: str
    field: str
    origin: str


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _recorded(payload: Any) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return the revisions ``payload`` records and every string leaf it holds.

    One traversal answers both questions a caller has about an artifact -- which
    revisions it records, and what it states beside them -- so an artifact is
    decoded once and walked once per reader of it.
    """

    found: dict[str, list[str]] = {}
    strings: dict[str, str] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        else:
            if isinstance(node, str):
                strings[path] = node
            if is_full_revision(node):
                found.setdefault(node, []).append(path)

    walk(payload, "")
    return {revision: tuple(paths) for revision, paths in found.items()}, strings


def recorded_revisions(payload: Any) -> dict[str, tuple[str, ...]]:
    """Map each full-length revision in ``payload`` to the paths that record it.

    Args:
        payload: Decoded JSON value of an evidence artifact.

    Returns:
        One entry per distinct revision, holding the dotted-and-indexed JSON paths
        that carry it in document order. Paths are reported so an error can name
        every place a revision appears instead of only the first.
    """

    revisions, _ = _recorded(payload)
    return revisions


def recorded_origins(payload: Any) -> tuple[RecordedOrigin, ...]:
    """Return the origins ``payload`` states beside the revisions it records.

    The convention is ``<field>_origin`` beside any field that records a revision,
    so ``source.revision_origin`` beside ``source.revision`` and
    ``base_commit_origin`` beside ``base_commit`` both count. An artifact that
    states its own origin is accounted for without the table, and one that states
    an origin contradicting the table is reported instead of being believed twice.

    Args:
        payload: Decoded JSON value of an evidence artifact.

    Returns:
        One entry per revision field that has a string ``_origin`` beside it.
    """

    return _stated_origins(*_recorded(payload))


def _stated_origins(
    revisions: Mapping[str, tuple[str, ...]], strings: Mapping[str, str]
) -> tuple[RecordedOrigin, ...]:
    """Pair each recorded revision field with the ``_origin`` field beside it."""

    stated: list[RecordedOrigin] = []
    for revision, paths in revisions.items():
        for path in paths:
            origin = strings.get(f"{path}_origin")
            if origin is not None:
                stated.append(
                    RecordedOrigin(revision=revision, field=path, origin=origin)
                )
    return tuple(stated)


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
    records: Mapping[str, Mapping[str, tuple[str, ...]]],
    *,
    root: Path = ROOT,
    evidence_roots: Sequence[Path] = EVIDENCE_ROOTS,
) -> tuple[str, ...]:
    """Return why a declared origin does not match the evidence it describes.

    Args:
        document: Decoded declaration table.
        records: Each walked artifact mapped to the revisions it records, so a
            declaration is checked against evidence the walk already decoded.
        root: Repository the declaration paths are relative to.
        evidence_roots: Directories the walk covered.
    """

    errors: list[str] = []
    for declaration in _declared_origins(document):
        target = root / declaration.artifact
        covered = [
            artifact
            for artifact in records
            if _declaration_covers(declaration.artifact, artifact)
        ]
        if not target.is_dir() and not target.is_file():
            errors.append(
                f"{declaration.artifact} is declared but is neither a file nor a "
                "directory in this repository"
            )
            continue
        if not any(
            _is_within(target, evidence_root) for evidence_root in evidence_roots
        ):
            roots = ", ".join(sorted(path.name for path in evidence_roots))
            errors.append(
                f"{declaration.artifact} is declared but is outside the walked "
                f"evidence roots ({roots}), so the declaration cannot take effect"
            )
            continue
        if not any(declaration.revision in records[artifact] for artifact in covered):
            if target.is_dir():
                errors.append(
                    f"no artifact under {declaration.artifact} records the declared "
                    f"revision {declaration.revision}"
                )
            else:
                errors.append(
                    f"{declaration.artifact} does not record the declared revision "
                    f"{declaration.revision}"
                )
    return tuple(errors)


def _artifact_errors(
    artifact: str,
    revisions: Mapping[str, tuple[str, ...]],
    origins: Sequence[RevisionOrigin],
    *,
    stated: Sequence[RecordedOrigin] = (),
    root: Path = ROOT,
    resolutions: dict[str, bool] | None = None,
    publications: dict[str, bool] | None = None,
) -> tuple[str, ...]:
    """Return why an artifact's recorded revisions are not accounted for.

    Args:
        artifact: Repository-relative path used to match declarations.
        revisions: Recorded revisions mapped to the JSON paths that carry them.
        origins: Declarations from :func:`_declared_origins`.
        stated: Origins the artifact states inside itself for those revisions.
        root: Repository to resolve revisions against.
        resolutions: Cache of the presence answer, to keep one ``git`` probe per
            distinct revision across the whole walk.
        publications: Cache of the reachability answer, filled only for revisions
            this repository holds.

    Returns:
        One diagnostic per revision this repository does not yield and that
        therefore has no origin recorded in either place, per origin stated for a
        revision this repository does obtain, and per revision whose two records
        contradict each other.
    """

    declared: dict[str, list[RevisionOrigin]] = {}
    for origin in origins:
        if _declaration_covers(origin.artifact, artifact):
            declared.setdefault(origin.revision, []).append(origin)
    disclosures: dict[str, list[RecordedOrigin]] = {}
    for record in stated:
        disclosures.setdefault(record.revision, []).append(record)

    supported = ", ".join(repr(value) for value in SUPPORTED_ORIGINS)
    errors: list[str] = []
    for revision, paths in revisions.items():
        where = ", ".join(paths)
        obtainable = _obtainable(revision, root, resolutions, publications)
        rows = declared.get(revision, [])
        records = disclosures.get(revision, [])

        for record in records:
            if record.origin not in SUPPORTED_ORIGINS:
                errors.append(
                    f"{artifact} states {record.origin!r} in {record.field}_origin for "
                    f"{revision}, which is not a supported origin; use one of "
                    f"{supported}"
                )
        stated_origins = {
            record.origin for record in records if record.origin in SUPPORTED_ORIGINS
        }
        table_origins = {row.origin for row in rows}
        if table_origins and stated_origins and table_origins != stated_origins:
            errors.append(
                f"{artifact} records {revision} ({where}) with origin "
                f"{sorted(table_origins)} in {DECLARATIONS.name} and origin "
                f"{sorted(stated_origins)} in its own fields, which disagree; one of "
                "the two is wrong"
            )

        accounted_at = _accounted_at(rows, records)
        if accounted_at is not None:
            if obtainable:
                errors.append(
                    f"{artifact} states an origin for {revision} ({where}) {accounted_at}, "
                    "but a clone that fetches this repository obtains that commit, so "
                    "the revision needs no origin"
                )
            continue
        if obtainable:
            continue
        if _resolves(revision, root, resolutions):
            errors.append(
                f"{artifact} records {revision} ({where}), which this checkout holds "
                "but no ref of this repository reaches, so a clone that fetches this "
                f"repository does not obtain it; declare it in {DECLARATIONS.name} "
                f"with origin {ORIGIN_UNREFERENCED_OBJECT!r}"
            )
            continue
        errors.append(
            f"{artifact} records {revision} ({where}), which this checkout does not "
            f"hold; declare it in {DECLARATIONS.name} with the origin that accounts "
            f"for it -- {ORIGIN_UNREFERENCED_OBJECT!r} when the remote still serves "
            f"the object by name, {ORIGIN_PRODUCING_HOST_HISTORY!r} when the run "
            f"happened in a history this repository does not contain, or "
            f"{ORIGIN_EXTERNAL_DEPENDENCY!r} when the revision is a commit of a "
            f"dependency -- or state {paths[0]}_origin beside the field that records "
            "it"
        )
    return tuple(errors)


def _accounted_at(
    rows: Sequence[RevisionOrigin], records: Sequence[RecordedOrigin]
) -> str | None:
    """Return where a revision's origin is recorded, or None when it is not.

    An origin that this module does not support does not account for a revision:
    it is reported on its own, and the revision stays unaccounted for so that the
    missing disclosure is not hidden behind an unusable one.
    """

    places: list[str] = []
    if any(row.origin in SUPPORTED_ORIGINS for row in rows):
        places.append(f"in {DECLARATIONS.name}")
    if any(record.origin in SUPPORTED_ORIGINS for record in records):
        places.append("in its own fields")
    return " and ".join(places) if places else None


def _read_artifacts(evidence_roots: Sequence[Path], root: Path) -> tuple[
    dict[str, dict[str, tuple[str, ...]]],
    dict[str, tuple[RecordedOrigin, ...]],
    list[str],
]:
    """Decode every walked artifact once, keyed by repository-relative path.

    Returns:
        The revisions each artifact records, the origins each states inside itself,
        and the artifacts that could not be decoded.
    """

    records: dict[str, dict[str, tuple[str, ...]]] = {}
    disclosures: dict[str, tuple[RecordedOrigin, ...]] = {}
    errors: list[str] = []
    for evidence_root in evidence_roots:
        for path in sorted(evidence_root.rglob("*.json")):
            artifact = _relative(path, root)
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{artifact} could not be read as JSON: {error}")
                continue
            revisions, strings = _recorded(payload)
            records[artifact] = revisions
            disclosures[artifact] = _stated_origins(revisions, strings)
    return records, disclosures, errors


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
    records, disclosures, read_errors = _read_artifacts(evidence_roots, root)
    errors.extend(read_errors)
    errors.extend(
        _declaration_consistency_errors(
            document, records, root=root, evidence_roots=evidence_roots
        )
    )
    origins = _declared_origins(document)
    resolutions: dict[str, bool] = {}
    publications: dict[str, bool] = {}
    for artifact, revisions in records.items():
        errors.extend(
            _artifact_errors(
                artifact,
                revisions,
                origins,
                stated=disclosures.get(artifact, ()),
                root=root,
                resolutions=resolutions,
                publications=publications,
            )
        )
    return tuple(errors)


def citation_errors(
    *,
    root: Path = ROOT,
    citation_roots: Sequence[Path] = CITATION_ROOTS,
) -> tuple[str, ...]:
    """Return every revision cited in a form a reader cannot expand.

    A shortened hexadecimal run is reported when it names exactly one revision the
    repository gives in full somewhere and the file citing it does not give that
    revision in full itself. The first condition is what separates a citation from
    the other hexadecimal-looking text a file can hold -- a task identifier, a date,
    a digest -- since such text prefixes no recorded revision. The second is the
    defect: a reader who follows the citation has no way to learn which commit the
    prefix names, and a revision that cannot be named cannot be obtained.

    A run of zeros is skipped. The null-commit placeholder is hexadecimal and names
    no commit by construction, so it is not a citation a reader could expand.
    """

    complete = _revisions_given_in_full(root)
    errors: list[str] = []
    for citation_root in citation_roots:
        for path in sorted(citation_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CITATION_SUFFIXES:
                continue
            if any(part in CITATION_SKIP_DIRECTORIES for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            local = {
                match.group(1).lower()
                for match in _FULL_REVISION_PATTERN.finditer(text)
            }
            for match in _SHORT_REVISION_PATTERN.finditer(text):
                cited = match.group(1).lower()
                if set(cited) == {"0"}:
                    continue
                targets = [value for value in complete if value.startswith(cited)]
                if len(targets) != 1 or targets[0] in local:
                    continue
                artifact = _relative(path, root)
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{artifact}:{line} cites {cited}, which names {targets[0]} in "
                    f"this repository; give that revision in full here, because a "
                    f"reader cannot obtain a revision the citation does not name"
                )
    return tuple(errors)


def _revisions_given_in_full(root: Path) -> frozenset[str]:
    """Return every revision the repository gives in full, at any length of file."""

    complete: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CITATION_SUFFIXES:
            continue
        if any(part in CITATION_SKIP_DIRECTORIES for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        complete.update(
            match.group(1).lower() for match in _FULL_REVISION_PATTERN.finditer(text)
        )
    return frozenset(complete)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, default=DECLARATIONS)
    args = parser.parse_args(argv)
    try:
        errors = evidence_errors(declarations=args.declarations)
    except ProvenanceUnavailableError as error:
        print(f"evidence revision origins could not be checked: {error}")
        return 1
    errors += citation_errors()
    if errors:
        print("\n".join(errors))
        return 1
    print("evidence revision origins passed")
    return 0


def _declaration_covers(declared: str, artifact: str) -> bool:
    """Return whether a declaration written for ``declared`` applies to ``artifact``.

    A declaration names one artifact or a directory, in which case it applies to
    every artifact below that directory, so a bulk evidence surface states one
    origin instead of repeating it for each file.
    """

    prefix = declared.rstrip("/")
    return artifact == prefix or artifact.startswith(f"{prefix}/")


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


def _published(revision: str, root: Path, publications: dict[str, bool] | None) -> bool:
    # Annotated for the same reason as in ``_resolves``.
    if publications is None:
        published: bool = revision_is_published(revision, root)
        return published
    if revision not in publications:
        publications[revision] = revision_is_published(revision, root)
    return publications[revision]


def _obtainable(
    revision: str,
    root: Path,
    resolutions: dict[str, bool] | None,
    publications: dict[str, bool] | None,
) -> bool:
    """Return whether a clone that fetches this repository obtains ``revision``.

    Presence and reachability are separate questions, and only both together mean
    a reader can obtain the commit. A revision that resolves only because this
    checkout still holds an object no ref reaches is not obtained by a clone or a
    plain fetch, so it needs the same disclosure as a revision this repository
    never contained. Asking reachability only after presence keeps a repository
    that holds nothing from being asked for the refs reaching it.
    """

    if not _resolves(revision, root, resolutions):
        return False
    return _published(revision, root, publications)


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
