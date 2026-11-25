"""Unsigned API source restoration must reproduce Git's exact object bytes."""

from __future__ import annotations

import copy
import subprocess
from datetime import datetime, timezone

import pytest

from tools.ci_source_archive import _verified_commit

pytestmark = pytest.mark.unit


def _source(
    author_zone: str, committer_zone: str, *, extra_header: str = ""
) -> tuple[dict[str, object], bytes, str]:
    message = "Source restoration\n\nKeep amplitude phase.\n"
    raw = (
        "tree " + "a" * 40 + "\n"
        "parent " + "b" * 40 + "\n"
        f"author FlagQuantum <liuwei.chem.phys@gmail.com> 1700000000 {author_zone}\n"
        f"committer FlagQuantum <liuwei.chem.phys@gmail.com> 1700000010 {committer_zone}\n"
        + extra_header
        + "\n"
        + message
    ).encode()
    sha = (
        subprocess.run(
            ["git", "hash-object", "-t", "commit", "--stdin"],
            input=raw,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    metadata: dict[str, object] = {
        "sha": sha,
        "tree": {"sha": "a" * 40},
        "parents": [{"sha": "b" * 40}],
        "author": {
            "name": "FlagQuantum",
            "email": "liuwei.chem.phys@gmail.com",
            "date": datetime.fromtimestamp(1700000000, timezone.utc).isoformat(),
        },
        "committer": {
            "name": "FlagQuantum",
            "email": "liuwei.chem.phys@gmail.com",
            "date": datetime.fromtimestamp(1700000010, timezone.utc).isoformat(),
        },
        "message": message.rstrip("\n"),
        "verification": {"payload": None, "signature": None, "reason": "unsigned"},
    }
    return metadata, raw, sha


@pytest.mark.parametrize(
    ("author_zone", "committer_zone"),
    [("+0800", "+0800"), ("+0000", "-0330"), ("-0700", "+1245"), ("-0000", "+0000")],
)
@pytest.mark.parametrize("keep_final_newline", [True, False])
def test_unsigned_metadata_recovers_only_the_exact_git_object(
    author_zone: str, committer_zone: str, keep_final_newline: bool
) -> None:
    metadata, raw, sha = _source(author_zone, committer_zone)
    if keep_final_newline:
        metadata["message"] = str(metadata["message"]) + "\n"
    actual = _verified_commit(metadata, sha)
    assert actual.sha == sha
    assert actual.raw == raw
    assert actual.tree == "a" * 40


@pytest.mark.parametrize("field", ["author", "committer", "message", "parents", "tree"])
def test_altered_unsigned_metadata_is_rejected(field: str) -> None:
    metadata, _, sha = _source("+0800", "+0800")
    altered = copy.deepcopy(metadata)
    if field in ("author", "committer"):
        actor = altered[field]
        assert isinstance(actor, dict)
        actor["email"] = "changed@example.com"
    elif field == "message":
        altered[field] = "Changed message"
    elif field == "parents":
        altered[field] = [{"sha": "c" * 40}]
    else:
        altered[field] = {"sha": "c" * 40}
    with pytest.raises(ValueError, match="cannot reproduce"):
        _verified_commit(altered, sha)


def test_unrecoverable_headers_keep_the_git_transport_refusal() -> None:
    metadata, _, sha = _source("+0800", "+0800", extra_header="encoding UTF-8\n")
    with pytest.raises(ValueError, match="Git transport.*cannot reproduce"):
        _verified_commit(metadata, sha)


@pytest.mark.parametrize(
    "change",
    [
        "actor",
        "fractional_date",
        "naive_date",
        "parent",
        "too_many_parents",
        "message_size",
    ],
)
def test_unsigned_metadata_budget_and_shape_checks(change: str) -> None:
    metadata, _, sha = _source("+0800", "+0800")
    if change == "actor":
        metadata["author"] = {"name": "FlagQuantum"}
    elif change in ("fractional_date", "naive_date"):
        actor = metadata["author"]
        assert isinstance(actor, dict)
        actor["date"] = (
            "2023-11-14T22:13:20.125Z"
            if change == "fractional_date"
            else "2023-11-14T22:13:20"
        )
    elif change == "parent":
        metadata["parents"] = [{"sha": "invalid"}]
    elif change == "too_many_parents":
        metadata["parents"] = [{"sha": "b" * 40}] * 17
    else:
        metadata["message"] = "x" * 16385
    with pytest.raises(ValueError):
        _verified_commit(metadata, sha)
