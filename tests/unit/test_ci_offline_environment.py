"""Locked CPU containers preserve isolation and job process settings."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name == "nt", reason="The CPU container wrapper requires bash")
@pytest.mark.parametrize("binary", ["python", "torchrun"])
@pytest.mark.parametrize("cached_wheels", [False, True])
def test_container_wrapper_preserves_offline_runtime_and_read_only_wheels(
    tmp_path: Path, binary: str, cached_wheels: bool
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "docker.json"
    docker = commands / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_DOCKER_LOG'], 'w') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n"
    )
    docker.chmod(0o755)
    wrapper = commands / binary
    wrapper.symlink_to(ROOT / ".github/actions/offline-cpu/run")
    workspace = tmp_path / "workspace with spaces"
    temporary = tmp_path / "temporary"
    wheels = tmp_path / "certified wheels"
    for directory in (workspace, temporary, wheels):
        directory.mkdir()
    private = temporary / "private"
    image = "ghcr.io/flagos-ai/flagquantum-ci@sha256:" + "1" * 64
    arguments = (
        ["--standalone", "--nproc-per-node=2"]
        if binary == "torchrun"
        else ["-c", "print('offline')"]
    )
    subprocess.run(
        ["bash", str(wrapper), *arguments],
        env={
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TEST_DOCKER_LOG": str(log),
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(temporary),
            "FQ_CI_PRIVATE": str(private),
            "FQ_CI_CONTAINER_USER": "0:0",
            "FQ_CI_IMAGE": image,
            "FQ_CI_WHEEL_DIRECTORY": str(wheels) if cached_wheels else "",
            "FQ_DISTRIBUTED_WORKERS": "1",
            "MAX_JOBS": "2",
        },
        check=True,
        capture_output=True,
        timeout=10,
    )
    actual = json.loads(log.read_text())
    assert actual[:2] == ["run", "--rm"]
    assert actual[actual.index("--network") + 1] == "none"
    assert actual[actual.index("--pull") + 1] == "never"
    assert "--read-only" in actual
    assert actual[actual.index("--user") + 1] == "0:0"
    mounts = [actual[i + 1] for i, value in enumerate(actual) if value == "--volume"]
    assert f"{workspace}:{workspace}" in mounts
    assert f"{temporary}:{temporary}" in mounts
    assert (f"{wheels}:{wheels}:ro" in mounts) == cached_wheels
    assert len(mounts) == (3 if cached_wheels else 2)
    environment = [actual[i + 1] for i, value in enumerate(actual) if value == "--env"]
    assert "FQ_DISTRIBUTED_WORKERS" in environment
    assert "MAX_JOBS" in environment
    assert "GLOO_SOCKET_IFNAME=lo" in environment
    command = actual[actual.index(image) + 1 :]
    expected = [str(private / "venv/bin/python")]
    if binary == "torchrun":
        expected.extend(["-m", "torch.distributed.run"])
    assert command == [*expected, *arguments]
