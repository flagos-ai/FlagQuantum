"""Build scratch files belong to the isolated Python job volume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("creation_fails", [False, True])
def test_python_build_temporary_directory_is_private_and_persisted(
    tmp_path: Path, creation_fails: bool
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/isolated-python/action.yml").read_text()
    )
    command = action["runs"]["steps"][0]["run"]
    executable = tmp_path / "bin"
    executable.mkdir()
    python = executable / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, tempfile\n"
        "with tempfile.NamedTemporaryFile() as scratch:\n"
        "    with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "        stream.write(json.dumps({'scratch': scratch.name, "
        "'tmpdir': os.environ.get('TMPDIR')}) + '\\n')\n"
    )
    python.chmod(0o755)
    if creation_fails:
        mktemp = executable / "mktemp"
        mktemp.write_text("#!/bin/sh\nexit 1\n")
        mktemp.chmod(0o755)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    shared = tmp_path / "shared-temp"
    shared.mkdir()
    sentinel = shared / "other-job"
    sentinel.write_text("keep")
    call_log = tmp_path / "calls.jsonl"
    job_directories: list[Path] = []
    for index in range(2):
        environment_file = tmp_path / f"environment-{index}"
        environment = {
            **os.environ,
            "PATH": f"{executable}{os.pathsep}{os.environ['PATH']}",
            "TMPDIR": str(shared),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(environment_file),
            "GITHUB_PATH": str(tmp_path / f"path-{index}"),
            "CALL_LOG": str(call_log),
            "RUNNER_ENVIRONMENT": "github-hosted",
            "FQ_RUNNER_KIND": "",
        }
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", command],
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if creation_fails:
            assert result.returncode != 0
            assert not call_log.exists()
            assert not environment_file.exists()
            continue
        assert result.returncode == 0, result.stderr
        recorded = json.loads(call_log.read_text().splitlines()[-1])
        job_directory = Path(recorded["tmpdir"])
        assert job_directory.parent == runner_temp
        assert Path(recorded["scratch"]).parent == job_directory
        assert f"TMPDIR={job_directory}" in environment_file.read_text().splitlines()
        job_directories.append(job_directory)
    assert sentinel.read_text() == "keep"
    if not creation_fails:
        assert len(set(job_directories)) == 2
