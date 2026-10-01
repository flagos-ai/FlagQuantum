"""A recorded evidence revision must be obtainable, or say why it is not.

Only six validators asked `tools/evidence_provenance.py` whether a recorded
revision names a real commit, and each one had its artifact path written into it.
An artifact that no validator named was never asked, so nineteen artifacts under
`artifacts/` recorded revisions this repository does not contain and thirteen of
them said nothing about it. The gate under test asks the same question of a
directory, and `evidence-revision-origins.toml` records the origin of every
revision the answer is no for.

These tests build their own repository and artifacts rather than reading the
ambient clone, so they state the gate's behaviour instead of the state of one
checkout. The last test reads the real tree, which is what makes the checked-in
disclosure a checked fact rather than a claim in a pull request.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.check_evidence_revisions import (
    ORIGIN_EXTERNAL_DEPENDENCY,
    evidence_errors,
    recorded_revisions,
)
from tools.evidence_provenance import (
    ORIGIN_PRODUCING_HOST_HISTORY,
    ProvenanceUnavailableError,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]

#: A full-length hexadecimal revision no repository contains.
ABSENT_REVISION = "0" * 40

#: A second absent revision, used where a declaration and an artifact disagree.
OTHER_ABSENT_REVISION = "1" * 40

DECLARATION_HEADER = 'schema = "flagquantum.evidence_revision_origins.v1"\n'


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A complete one-commit repository with an empty ``artifacts`` directory."""

    root = tmp_path / "repository"
    (root / "artifacts").mkdir(parents=True)
    _git(root, "init", "-q", "--initial-branch=main")
    _git(
        root,
        "-c",
        "user.email=evidence@example.invalid",
        "-c",
        "user.name=Evidence",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "initial",
    )
    return root


def _write_artifact(root: Path, name: str, payload: Any) -> Path:
    path = root / "artifacts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_declarations(root: Path, body: str) -> Path:
    path = root / "evidence-revision-origins.toml"
    path.write_text(DECLARATION_HEADER + body, encoding="utf-8")
    return path


def _declaration(
    path: str, revision: str, origin: str, repository: str | None = None
) -> str:
    lines = [
        "[[artifact]]",
        f'path = "{path}"',
        "[[artifact.revision]]",
        f'value = "{revision}"',
        f'origin = "{origin}"',
    ]
    if repository is not None:
        lines.append(f'repository = "{repository}"')
    return "\n".join(lines) + "\n"


def _errors(root: Path) -> tuple[str, ...]:
    return evidence_errors(
        root=root,
        declarations=root / "evidence-revision-origins.toml",
        evidence_roots=(root / "artifacts",),
    )


def test_undeclared_unresolvable_revision_fails_the_gate(repository: Path) -> None:
    # Red before the gate existed: the six checked-in validators never read these
    # artifacts, so this pin passed every check in the repository.
    _write_artifact(
        repository, "validated.json", {"source_revision": OTHER_ABSENT_REVISION}
    )
    _write_artifact(
        repository,
        "unvalidated.json",
        {"environment": {"source_revision": ABSENT_REVISION}},
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/validated.json",
            OTHER_ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "artifacts/unvalidated.json" in errors[0]
    assert ABSENT_REVISION in errors[0]


def test_declared_unresolvable_revision_passes(repository: Path) -> None:
    # The control for the test above: the same artifact passes once its origin is
    # recorded, so the failure is the missing disclosure and nothing else.
    _write_artifact(
        repository,
        "unvalidated.json",
        {"environment": {"source_revision": ABSENT_REVISION}},
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/unvalidated.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )

    assert _errors(repository) == ()


def test_gate_walks_artifacts_no_validator_names(repository: Path) -> None:
    # Coverage is the point of the change: an artifact added under the walked root
    # is checked without anyone editing the gate, which is what the six hardcoded
    # validators could not do.
    _write_declarations(
        repository,
        _declaration(
            "artifacts/known.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )
    _write_artifact(repository, "known.json", {"source_revision": ABSENT_REVISION})
    assert _errors(repository) == ()

    _write_artifact(
        repository, "added_later.json", {"source_revision": OTHER_ABSENT_REVISION}
    )

    errors = _errors(repository)
    assert len(errors) == 1
    assert "artifacts/added_later.json" in errors[0]


def test_declaring_a_revision_this_repository_contains_fails(repository: Path) -> None:
    # An origin is a disclosure that the revision is unavailable here. Declaring one
    # for a revision that resolves hides a checkable pin behind a claim, so it is
    # rejected rather than tolerated.
    head = _git(repository, "rev-parse", "HEAD")
    _write_artifact(repository, "resolving.json", {"source_revision": head})
    _write_declarations(
        repository,
        _declaration("artifacts/resolving.json", head, ORIGIN_PRODUCING_HOST_HISTORY),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "needs no origin" in errors[0]


def test_declaring_a_revision_the_artifact_does_not_record_fails(
    repository: Path,
) -> None:
    # A declaration that nothing backs is a record a reader cannot use, and it also
    # means the artifact's own pin was never disclosed.
    _write_artifact(repository, "recorded.json", {"source_revision": ABSENT_REVISION})
    _write_declarations(
        repository,
        _declaration(
            "artifacts/recorded.json",
            OTHER_ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 2
    assert any("does not record the declared revision" in error for error in errors)
    assert any(
        ABSENT_REVISION in error and "names no commit" in error for error in errors
    )


def test_declaration_outside_the_walked_roots_fails(repository: Path) -> None:
    # A declaration the walk never consults reads as disclosure while checking
    # nothing, which is the failure mode this gate exists to remove.
    (repository / "benchmarks").mkdir()
    (repository / "benchmarks" / "results.json").write_text("{}\n", encoding="utf-8")
    _write_declarations(
        repository,
        _declaration(
            "benchmarks/results.json",
            ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "outside the walked" in errors[0]


def test_declaration_for_a_missing_artifact_fails(repository: Path) -> None:
    _write_declarations(
        repository,
        _declaration(
            "artifacts/absent.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "not a file in this repository" in errors[0]


def test_missing_declaration_file_fails_closed(repository: Path) -> None:
    _write_artifact(
        repository, "unvalidated.json", {"source_revision": ABSENT_REVISION}
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "is missing" in errors[0]


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ("", "must declare at least one artifact"),
        (
            _declaration("artifacts/a.json", "0" * 39, ORIGIN_PRODUCING_HOST_HISTORY),
            "requires a full hexadecimal revision",
        ),
        (
            _declaration("artifacts/a.json", ABSENT_REVISION, "somewhere_else"),
            "is not supported",
        ),
        (
            _declaration(
                "artifacts/a.json", ABSENT_REVISION, ORIGIN_EXTERNAL_DEPENDENCY
            ),
            "requires a 'repository'",
        ),
        (
            _declaration(
                "artifacts/a.json",
                ABSENT_REVISION,
                ORIGIN_PRODUCING_HOST_HISTORY,
                repository="flagos-ai/Torch-FL",
            ),
            "applies only to",
        ),
        (
            _declaration(
                "artifacts/a.json",
                ABSENT_REVISION,
                ORIGIN_EXTERNAL_DEPENDENCY,
                "not-a-repo",
            ),
            "requires a 'repository'",
        ),
    ),
)
def test_malformed_declarations_are_rejected(
    repository: Path, body: str, expected: str
) -> None:
    _write_declarations(repository, body)

    errors = _errors(repository)

    assert errors, body
    assert any(expected in error for error in errors), errors


def test_duplicate_declarations_are_rejected(repository: Path) -> None:
    _write_declarations(
        repository,
        _declaration("artifacts/a.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY)
        + _declaration(
            "artifacts/a.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )

    errors = _errors(repository)

    assert any("more than once" in error for error in errors), errors


def test_recorded_revisions_reports_every_location(repository: Path) -> None:
    payload = {
        "environment": {"source_revision": ABSENT_REVISION},
        "runs": [
            {"environment": {"source_revision": ABSENT_REVISION}},
            {"environment": {"torch_fl_revision": OTHER_ABSENT_REVISION}},
        ],
        "notes": ["not a revision: " + "z" * 40],
        "abbreviated": "0" * 39,
    }

    recorded = recorded_revisions(payload)

    assert recorded[ABSENT_REVISION] == (
        "environment.source_revision",
        "runs[0].environment.source_revision",
    )
    assert recorded[OTHER_ABSENT_REVISION] == ("runs[1].environment.torch_fl_revision",)


def test_every_undeclared_revision_is_named_in_the_failure(repository: Path) -> None:
    _write_artifact(
        repository,
        "two.json",
        {
            "torch_fl_revision": ABSENT_REVISION,
            "runs": [{"source_revision": OTHER_ABSENT_REVISION}],
        },
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/two.json", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert OTHER_ABSENT_REVISION in errors[0]
    assert "runs[0].source_revision" in errors[0]


def test_checked_in_evidence_accounts_for_every_recorded_revision() -> None:
    # The green-after state of this change, asserted against the real tree: every
    # full-length revision recorded under `artifacts/` either resolves against this
    # repository or is declared in `evidence-revision-origins.toml`. This is what
    # fails when a new artifact records a pin nobody can obtain, so the disclosure
    # cannot quietly fall behind the evidence again.
    try:
        errors = evidence_errors()
    except ProvenanceUnavailableError as error:  # pragma: no cover - deep-clone lanes
        pytest.skip(f"this checkout cannot resolve revisions: {error}")

    assert errors == ()
