"""A recorded source revision is evidence only if someone can obtain it.

The evidence validators used to accept any 40-character string as a source
revision, so a revision that named no commit passed exactly like the commit it
claimed to be. These tests hold the two halves of the repair together: the
shared resolver distinguishes "absent from this repository" from "this
repository cannot answer", and every artifact the gates validate either resolves
or discloses why it cannot.

The resolver tests build their own repository instead of using the ambient
clone, which may be a depth-1 checkout that cannot distinguish the two.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.evidence_provenance import (
    ORIGIN_PRODUCING_HOST_HISTORY,
    ProvenanceUnavailableError,
    is_full_revision,
    revision_resolves,
    source_revision_errors,
)
from tools.validate_flagos_reference_evidence import (
    evidence_errors as flagos_reference_errors,
)
from tools.validate_split_real_imag_device_double_single_evidence import (
    evidence_errors as split_p4_errors,
)
from tools.validate_split_real_imag_double_single_evidence import (
    evidence_errors as split_p3_errors,
)
from tools.validate_split_real_imag_optimizer_evidence import (
    evidence_errors as split_p5_errors,
)
from tools.validate_split_real_imag_precision_evidence import (
    evidence_errors as split_p2_errors,
)
from tools.validate_split_real_imag_training_evidence import (
    evidence_errors as split_p1_errors,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
LABEL = "test evidence"

#: A full-length hexadecimal revision no repository contains.
ABSENT_REVISION = "0" * 40

#: Every artifact a checked-in gate validates, with the validator that gates it.
GATED_ARTIFACTS: tuple[tuple[str, Callable[[dict], tuple[str, ...]]], ...] = (
    ("artifacts/flagos_cuda_reference_a800_20260824.json", flagos_reference_errors),
    ("artifacts/split_real_imag_training_a800_20260824.json", split_p1_errors),
    ("artifacts/split_real_imag_precision_a800_20260824.json", split_p2_errors),
    ("artifacts/split_real_imag_double_single_a800_20260824.json", split_p3_errors),
    (
        "artifacts/split_real_imag_device_double_single_a800_20260825.json",
        split_p4_errors,
    ),
    ("artifacts/split_real_imag_optimizer_a800_20260825.json", split_p5_errors),
)


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
    """A complete one-commit repository, independent of the ambient clone."""

    root = tmp_path / "repository"
    root.mkdir()
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


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("0" * 40, True),
        ("f" * 40, True),
        ("0" * 39, False),
        ("0" * 41, False),
        ("0" * 39 + "z", False),
        ("F" * 40, False),
        ("unavailable", False),
        ("", False),
        (None, False),
        (40, False),
    ),
)
def test_only_a_full_hexadecimal_revision_is_a_revision(
    value: object, expected: bool
) -> None:
    assert is_full_revision(value) is expected


def test_a_recorded_revision_that_history_contains_resolves(repository: Path) -> None:
    head = _git(repository, "rev-parse", "HEAD")

    assert revision_resolves(head, repository) is True
    assert (
        source_revision_errors(
            {"revision": head, "tree_dirty": False}, label=LABEL, root=repository
        )
        == ()
    )


def test_a_recorded_revision_that_history_lacks_is_reported(
    repository: Path,
) -> None:
    assert revision_resolves(ABSENT_REVISION, repository) is False

    errors = source_revision_errors(
        {"revision": ABSENT_REVISION, "tree_dirty": False},
        label=LABEL,
        root=repository,
    )

    assert len(errors) == 1
    assert ABSENT_REVISION in errors[0]
    assert ORIGIN_PRODUCING_HOST_HISTORY in errors[0]


def test_a_disclosed_producing_host_revision_is_accepted(repository: Path) -> None:
    assert (
        source_revision_errors(
            {
                "revision": ABSENT_REVISION,
                "revision_origin": ORIGIN_PRODUCING_HOST_HISTORY,
                "tree_dirty": False,
            },
            label=LABEL,
            root=repository,
        )
        == ()
    )


def test_disclosure_does_not_excuse_a_value_that_is_not_a_revision(
    repository: Path,
) -> None:
    errors = source_revision_errors(
        {"revision": "unavailable", "revision_origin": ORIGIN_PRODUCING_HOST_HISTORY},
        label=LABEL,
        root=repository,
    )

    assert len(errors) == 1
    assert "full hexadecimal source revision" in errors[0]


def test_an_unsupported_origin_is_reported(repository: Path) -> None:
    errors = source_revision_errors(
        {"revision": ABSENT_REVISION, "revision_origin": "somewhere_else"},
        label=LABEL,
        root=repository,
    )

    assert len(errors) == 1
    assert "revision_origin" in errors[0]
    assert "somewhere_else" in errors[0]


def test_a_missing_source_identity_is_reported() -> None:
    assert source_revision_errors(None, label=LABEL) == (
        f"{LABEL} source identity is missing",
    )


def test_a_repository_that_cannot_answer_fails_closed(tmp_path: Path) -> None:
    # Not a repository at all: the resolver must say so rather than report that
    # the revision is absent, which would turn an unusable checkout into a pass.
    with pytest.raises(ProvenanceUnavailableError):
        revision_resolves(ABSENT_REVISION, tmp_path)

    errors = source_revision_errors(
        {"revision": ABSENT_REVISION, "tree_dirty": False},
        label=LABEL,
        root=tmp_path,
    )

    assert len(errors) == 1
    assert "could not be checked" in errors[0]


@pytest.mark.parametrize(("artifact", "_errors"), GATED_ARTIFACTS)
def test_every_gated_artifact_discloses_why_its_pin_cannot_resolve(
    artifact: str, _errors: Callable[[dict], tuple[str, ...]]
) -> None:
    source = json.loads((ROOT / artifact).read_text(encoding="utf-8"))["source"]

    assert source["revision_origin"] == ORIGIN_PRODUCING_HOST_HISTORY
    assert source["tree_dirty"] is False


@pytest.mark.parametrize(("artifact", "errors_for"), GATED_ARTIFACTS)
def test_each_gated_validator_rejects_an_undisclosed_pin(
    artifact: str, errors_for: Callable[[dict], tuple[str, ...]]
) -> None:
    payload = json.loads((ROOT / artifact).read_text(encoding="utf-8"))
    payload["source"].pop("revision_origin")

    assert errors_for(payload), artifact


@pytest.mark.parametrize(("artifact", "errors_for"), GATED_ARTIFACTS)
def test_each_gated_validator_accepts_a_disclosure_it_cannot_verify(
    artifact: str, errors_for: Callable[[dict], tuple[str, ...]]
) -> None:
    # The control for the test above: the same payload passes once the artifact
    # states why its pin is not resolvable here, so the failure is the missing
    # disclosure and not an unrelated complaint.
    assert errors_for(json.loads((ROOT / artifact).read_text(encoding="utf-8"))) == ()
