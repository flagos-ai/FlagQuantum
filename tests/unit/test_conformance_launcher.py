"""A failed or stalled placement rank must not leave other workers running."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import mps_placement_conformance as placement
from benchmarks.internal.evidence.representation_conformance import source_record

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("has_base", [False, True])
def test_source_identity_remains_truthful_without_a_remote_base(
    has_base: bool, tmp_path: Path
) -> None:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()

    git("init")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    package = tmp_path / "flagquantum"
    package.mkdir()
    source = package / "example.py"
    source.write_text("value = 1\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    head = git("rev-parse", "HEAD")
    if has_base:
        git("update-ref", "refs/remotes/origin/main", head)
    else:
        (tmp_path / ".git/shallow").write_text(head + "\n")
    record = source_record(tmp_path, ["benchmarks/example.py"])
    assert record["source_commit"] == head
    assert record["measuring_source_committed"] is True
    assert record["base_commit"] == (head if has_base else None)
    assert (record["base_commit_unavailable_reason"] is None) is has_base
    source.write_text("value = 2\n")
    assert (
        source_record(tmp_path, ["benchmarks/example.py"])["measuring_source_committed"]
        is False
    )


@pytest.mark.parametrize("failure", ["stall", "exit"])
def test_local_launcher_cleans_up_workers_on_failure(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def faulty_worker(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        # One failed rank must also stop the other rank, which remains asleep.
        program = "import time; time.sleep(30)"
        if failure == "exit" and not children:
            program = "raise SystemExit(4)"
        child = spawn([sys.executable, "-c", program], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(placement.subprocess, "Popen", faulty_worker)
    error = TimeoutError if failure == "stall" else RuntimeError
    with pytest.raises(error):
        placement.launch_local_world(
            2, 0.5 if failure == "stall" else 10, tmp_path / "failed.json", 11
        )
    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    assert len(list(tmp_path.glob("*.log"))) == 2
    assert not list(tmp_path.glob("*.json"))
