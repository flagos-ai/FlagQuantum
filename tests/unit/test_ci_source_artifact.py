"""Round-trip the exact source scripts the A800 jobs execute."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.name == "nt", reason="A800 source restoration runs in Linux bash"
)
@pytest.mark.parametrize(
    "workflow,job", [("ci.yml", "cpu-runtime"), ("local-gpu.yml", "one-gpu-local")]
)
def test_source_artifact_restores_the_exact_tree_and_rejects_another_commit(
    tmp_path: Path, workflow: str, job: str
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    pack = next(
        step["run"] for step in jobs["prepare-source"]["steps"] if "run" in step
    )
    restore = next(step["run"] for step in jobs[job]["steps"] if "run" in step)
    source = tmp_path / "source"
    target = tmp_path / "target"
    temporary = tmp_path / "temporary"
    for directory in (source, target, temporary):
        directory.mkdir()
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "RUNNER_TEMP": str(temporary),
    }

    def git(*arguments: str, cwd: Path = source) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()

    def shell(
        script: str, cwd: Path, *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-eo", "pipefail", "-c", script],
            cwd=cwd,
            env=environment,
            check=check,
            capture_output=True,
            text=True,
            timeout=20,
        )

    git("init")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    (source / "tracked.txt").write_text("parent\n")
    git("add", ".")
    git("commit", "-m", "parent")
    parent = git("rev-parse", "HEAD")
    (source / "tracked.txt").write_text("current\n")
    (source / "binary.dat").write_bytes(bytes(range(256)))
    (source / "link").symlink_to("tracked.txt")
    git("add", ".")
    git("commit", "-m", "current")
    sha = git("rev-parse", "HEAD")
    environment["GITHUB_SHA"] = sha
    shell(pack, source)

    for attempt in range(2):
        if attempt:
            git("sparse-checkout", "set", "--no-cone", "tracked.txt", cwd=target)
            (target / "tracked.txt").write_text("dirty\n")
            (target / "leftover.txt").write_text("stale\n")
        shell(restore, target)
        assert git("rev-parse", "HEAD", cwd=target) == sha
        assert git("status", "--porcelain", cwd=target) == ""
        assert (target / "tracked.txt").read_text() == "current\n"
        assert (target / "binary.dat").read_bytes() == bytes(range(256))
        assert (target / "link").is_symlink()
        assert not (target / "leftover.txt").exists()

    (temporary / "ci-source/source.sha").write_text(parent + "\n")
    assert shell(restore, target, check=False).returncode != 0
    assert git("rev-parse", "HEAD", cwd=target) == sha
