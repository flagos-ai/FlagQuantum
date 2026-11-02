"""A failed or stalled placement rank must not leave other workers running."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import mps_placement_conformance as placement

pytestmark = pytest.mark.unit


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
