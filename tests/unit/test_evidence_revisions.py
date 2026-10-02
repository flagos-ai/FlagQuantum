"""A recorded evidence revision must be obtainable, or say why it is not.

Only six validators asked `tools/evidence_provenance.py` whether a recorded
revision names a real commit, and each one had its artifact path written into it.
An artifact that no validator named was never asked, so nineteen artifacts under
`artifacts/` recorded revisions this repository does not contain and thirteen of
them said nothing about it. The gate under test asks a directory both questions a
reader depends on: whether this repository holds the commit, and whether a ref of
this repository reaches it. `evidence-revision-origins.toml` records the origin of
every revision the answer is no for.

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
    EVIDENCE_ROOTS,
    ORIGIN_EXTERNAL_DEPENDENCY,
    ORIGIN_UNREFERENCED_OBJECT,
    evidence_errors,
    recorded_origins,
    recorded_revisions,
)
from tools.evidence_provenance import (
    ORIGIN_PRODUCING_HOST_HISTORY,
    SUPPORTED_REVISION_ORIGINS,
    ProvenanceUnavailableError,
    revision_is_published,
    revision_resolves,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]

#: A full-length hexadecimal revision no repository contains.
ABSENT_REVISION = "0" * 40

#: A second absent revision, used where a declaration and an artifact disagree.
OTHER_ABSENT_REVISION = "1" * 40

#: The revision `_with_filler` declares, which no case under test mentions.
FILLER_REVISION = "9" * 40

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


def _unreferenced_commit(root: Path) -> str:
    """Add a commit to the object database that no ref reaches."""

    _git(root, "checkout", "-q", "-b", "side")
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
        "side",
    )
    revision = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    _git(root, "branch", "-q", "-D", "side")
    return revision


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


def _errors(root: Path, *evidence_roots: Path) -> tuple[str, ...]:
    return evidence_errors(
        root=root,
        declarations=root / "evidence-revision-origins.toml",
        evidence_roots=evidence_roots or (root / "artifacts",),
    )


def _with_filler(root: Path, body: str = "") -> None:
    """Write a well-formed table holding one unrelated declaration.

    ``evidence_errors`` rejects an empty table before it walks, so a case about an
    artifact that accounts for itself still needs the table to be usable. The filler
    records a revision no case mentions, so it contributes no diagnostic of its own.
    """

    _write_artifact(root, "filler.json", {"source_revision": FILLER_REVISION})
    _write_declarations(
        root,
        _declaration(
            "artifacts/filler.json", FILLER_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        )
        + body,
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


def test_gate_walks_the_benchmark_result_root(repository: Path) -> None:
    # Red before this change: the walked roots named `artifacts/` only, so the pins
    # under `benchmarks/results/` were never asked however many validators existed.
    # That is where the release-candidate artifacts recording revisions no ref
    # reaches were found, so the second root is asserted, not assumed.
    path = repository / "benchmarks" / "results" / "smoke" / "candidate.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"source_commit": ABSENT_REVISION}) + "\n", encoding="utf-8"
    )
    _write_artifact(
        repository, "declared.json", {"source_revision": OTHER_ABSENT_REVISION}
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/declared.json",
            OTHER_ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(
        repository,
        repository / "artifacts",
        repository / "benchmarks" / "results",
    )

    assert len(errors) == 1
    assert "benchmarks/results/smoke/candidate.json" in errors[0]
    assert "source_commit" in errors[0]


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


def test_undeclared_revision_no_ref_reaches_fails(repository: Path) -> None:
    # Red before this change: the gate asked only whether the object was in the
    # database. A checkout that once held a branch keeps its commits, so a pin like
    # this resolved on the machine that produced it and was unobtainable for every
    # reader, which is the same defect as a pin no repository contains.
    revision = _unreferenced_commit(repository)
    assert revision_resolves(revision, repository) is True
    assert revision_is_published(revision, repository) is False

    _write_artifact(repository, "orphaned.json", {"source_revision": revision})
    # The gate reads a missing table as an error before the walk, so the table
    # carries one well-formed declaration and the orphan is what is left over.
    _write_artifact(
        repository, "declared.json", {"source_revision": OTHER_ABSENT_REVISION}
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/declared.json",
            OTHER_ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "no ref of this repository reaches" in errors[0]
    assert ORIGIN_UNREFERENCED_OBJECT in errors[0]
    assert "names no commit" not in errors[0]


def test_declared_revision_no_ref_reaches_passes(repository: Path) -> None:
    # The control for the test above: the failure is the missing disclosure and
    # nothing else.
    revision = _unreferenced_commit(repository)
    _write_artifact(repository, "orphaned.json", {"source_revision": revision})
    _write_declarations(
        repository,
        _declaration("artifacts/orphaned.json", revision, ORIGIN_UNREFERENCED_OBJECT),
    )

    assert _errors(repository) == ()


def test_origin_stated_inside_the_artifact_accounts_for_it(repository: Path) -> None:
    # Red before this change: the artifact disclosed why its pin is not obtainable
    # here, `tools/evidence_provenance.py` read that disclosure, and this gate
    # ignored it and demanded a row in the table anyway -- two records of one fact,
    # neither aware of the other. Six checked-in artifacts state their own origin,
    # so the disagreement was reachable on real evidence.
    _with_filler(repository)
    _write_artifact(
        repository,
        "self_described.json",
        {
            "source": {
                "revision": ABSENT_REVISION,
                "revision_origin": "producing_host_history",
            }
        },
    )

    assert _errors(repository) == ()


def test_origin_stated_beside_an_ordinary_field_is_honoured(repository: Path) -> None:
    # The convention generalizes by suffix: whatever field records the revision
    # carries the origin, so an artifact recording `base_commit` states
    # `base_commit_origin` rather than reaching for a fixed key name.
    _with_filler(repository)
    _write_artifact(
        repository,
        "handoff.json",
        {
            "base_commit": ABSENT_REVISION,
            "base_commit_origin": "producing_host_history",
        },
    )

    assert _errors(repository) == ()


def test_origin_stated_inside_the_artifact_and_in_the_table_must_agree(
    repository: Path,
) -> None:
    # Green before this change, which is the defect: the two records were never
    # compared, so an artifact could contradict the table and pass twice. A record
    # that contradicts itself is worse than one that says nothing, because it reads
    # as a disclosure.
    _with_filler(repository)
    _write_artifact(
        repository,
        "contradicting.json",
        {
            "source": {
                "revision": ABSENT_REVISION,
                "revision_origin": "external_dependency",
            }
        },
    )
    _write_declarations(
        repository,
        _declaration(
            "artifacts/filler.json", FILLER_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        )
        + _declaration(
            "artifacts/contradicting.json",
            ABSENT_REVISION,
            ORIGIN_PRODUCING_HOST_HISTORY,
        ),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "disagree" in errors[0]
    assert ABSENT_REVISION in errors[0]


def test_origin_stated_for_an_obtainable_revision_fails(repository: Path) -> None:
    # The rule that makes an origin a disclosure rather than a hiding place applies
    # to the artifact's own record too: a revision a clone obtains needs no origin,
    # so stating one for it is rejected wherever it is stated.
    _with_filler(repository)
    head = _git(repository, "rev-parse", "HEAD")
    _write_artifact(
        repository,
        "resolving.json",
        {"source": {"revision": head, "revision_origin": "producing_host_history"}},
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "needs no origin" in errors[0]


def test_unsupported_origin_stated_inside_the_artifact_is_rejected(
    repository: Path,
) -> None:
    # An unusable origin is named as such and does not account for the revision, so
    # a typo cannot substitute for a disclosure.
    _with_filler(repository)
    _write_artifact(
        repository,
        "mistyped.json",
        {"base_commit": ABSENT_REVISION, "base_commit_origin": "somewhere_else"},
    )

    errors = _errors(repository)

    assert len(errors) == 2
    assert any("is not a supported origin" in error for error in errors), errors
    assert any("names no commit" in error for error in errors), errors


def test_directory_declaration_covers_every_artifact_below_it(repository: Path) -> None:
    # A bulk evidence surface states one origin once instead of repeating it for
    # every file, which is how the thirty-one artifacts recording one revision are
    # disclosed without thirty-one identical blocks.
    _write_artifact(repository, "suite/first.json", {"source_commit": ABSENT_REVISION})
    _write_artifact(repository, "suite/second.json", {"source_commit": ABSENT_REVISION})
    _write_declarations(
        repository,
        _declaration("artifacts/suite", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY),
    )

    assert _errors(repository) == ()


def test_directory_declaration_stops_at_the_directory_boundary(
    repository: Path,
) -> None:
    # A prefix is not a directory. `artifacts/suite` must not silently cover
    # `artifacts/suite2`, or a declaration would read as disclosing evidence it
    # never mentions.
    _write_artifact(repository, "suite/first.json", {"source_commit": ABSENT_REVISION})
    _write_artifact(
        repository, "suite2/second.json", {"source_commit": ABSENT_REVISION}
    )
    _write_declarations(
        repository,
        _declaration("artifacts/suite", ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY),
    )

    errors = _errors(repository)

    assert len(errors) == 1
    assert "artifacts/suite2/second.json" in errors[0]


def test_directory_declaration_without_a_matching_artifact_fails(
    repository: Path,
) -> None:
    _write_artifact(repository, "suite/first.json", {"source_commit": ABSENT_REVISION})
    _write_declarations(
        repository,
        _declaration(
            "artifacts/suite", OTHER_ABSENT_REVISION, ORIGIN_PRODUCING_HOST_HISTORY
        ),
    )

    errors = _errors(repository)

    assert any(
        "no artifact under artifacts/suite records" in error for error in errors
    ), errors


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
    assert "neither a file nor a directory" in errors[0]


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
    # full-length revision recorded under `artifacts/` or `benchmarks/results/` is
    # either obtainable from this repository or declared in
    # `evidence-revision-origins.toml`. This is what fails when a new artifact
    # records a pin nobody can obtain, so the disclosure cannot quietly fall behind
    # the evidence again.
    try:
        errors = evidence_errors()
    except ProvenanceUnavailableError as error:  # pragma: no cover - deep-clone lanes
        pytest.skip(f"this checkout cannot resolve revisions: {error}")

    assert errors == ()


def test_checked_in_artifacts_state_origins_the_gate_reads() -> None:
    # The convention is only a checked fact if the real tree uses it, so this reads
    # both walked roots and requires the gate's second input to be populated. A tree
    # in which no artifact stated its own origin would leave the cross-check
    # unreachable, with the gate passing while comparing nothing. Agreement between
    # the two records is enforced on this same tree by the test above, which runs the
    # gate over it.
    #
    # The vocabulary is asserted too: an origin written inside an artifact is read by
    # `tools/evidence_provenance.py` as well, so it must be one that module supports.
    # `producing_host_history` is the only value the two vocabularies share, and the
    # only one any checked-in artifact states.
    stated: list[tuple[str, str, str]] = []
    for evidence_root in EVIDENCE_ROOTS:
        for path in sorted(evidence_root.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact = path.relative_to(ROOT).as_posix()
            stated.extend(
                (artifact, record.revision, record.origin)
                for record in recorded_origins(payload)
            )

    assert stated, "no checked-in artifact states the origin of a revision it records"
    for artifact, revision, origin in stated:
        assert origin in SUPPORTED_REVISION_ORIGINS, (artifact, revision, origin)
