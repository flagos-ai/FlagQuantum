"""Locked CPU containers preserve isolation and job process settings."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name == "nt", reason="The CPU container wrapper requires bash")
@pytest.mark.parametrize("binary", ["python", "torchrun"])
@pytest.mark.parametrize("cached_wheels", [False, True])
@pytest.mark.parametrize("host_budget", [False, True])
def test_container_wrapper_preserves_offline_runtime_and_read_only_wheels(
    tmp_path: Path, binary: str, cached_wheels: bool, host_budget: bool
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
    budget = temporary / "host-budget.json"
    budget.write_text(
        json.dumps(
            {
                "effective_cpus": 4,
                "memory_limit_bytes": 16 * 1024**3,
                "affinity": [4, 5, 84, 85],
                "shared_registrations_budgeted": 1,
            }
        )
    )
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
            "FQ_CI_HOST_BUDGET": str(budget) if host_budget else "",
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
    if host_budget:
        assert actual[actual.index("--cpus") + 1] == "4.0"
        assert actual[actual.index("--memory") + 1] == str(16 * 1024**3)
        assert actual[actual.index("--cpuset-cpus") + 1] == "4,5,84,85"
    else:
        assert "--cpus" not in actual
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


@pytest.mark.skipif(
    os.name == "nt", reason="Linux container cancellation uses POSIX signals"
)
def test_cancelled_wrapper_removes_exact_container_and_preserves_failure(tmp_path):
    import signal
    import sys
    import time

    commands = tmp_path / "commands"
    commands.mkdir()
    docker = commands / "docker"
    log = tmp_path / "operations"
    docker.write_text(
        "#!" + sys.executable + "\n"
        "import os, pathlib, signal, sys, time\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['OPERATIONS'], 'a') as f: f.write(args[0] + '\\n')\n"
        "if args[0] == 'run':\n"
        "    pathlib.Path(args[args.index('--cidfile') + 1]).write_text('a' * 64)\n"
        "    time.sleep(30)\n"
        "elif args[0] == 'rm':\n"
        "    assert args == ['rm', '--force', 'a' * 64]\n"
    )
    docker.chmod(0o755)
    wrapper = commands / "python"
    wrapper.symlink_to(ROOT / ".github/actions/offline-cpu/run")
    private = tmp_path / "private"
    private.mkdir()
    process = subprocess.Popen(
        ["bash", str(wrapper), "-c", "pass"],
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "OPERATIONS": str(log),
            "GITHUB_WORKSPACE": str(tmp_path),
            "RUNNER_TEMP": str(tmp_path),
            "FQ_CI_PRIVATE": str(private),
            "FQ_CI_HOST_BUDGET": "",
            "FQ_CI_CONTAINER_USER": "0:0",
            "FQ_CI_IMAGE": "fixture",
            "FQ_CI_WHEEL_DIRECTORY": "",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not any(path.stat().st_size for path in private.glob("*.cid")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, (stdout, stderr)
        assert log.read_text().splitlines() == ["run", "rm"]
        assert not list(private.glob("*.cid"))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="The CPU environment requires bash")
@pytest.mark.parametrize(
    "image_present,imported", [(False, False), (True, False), (True, True)]
)
def test_provisioned_environment_never_logs_in_or_pulls(
    tmp_path: Path, image_present: bool, imported: bool
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "docker.jsonl"
    docker = commands / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_DOCKER_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    present = os.environ['TEST_IMAGE_PRESENT'] == 'true'\n"
        "    imported = os.environ['TEST_IMPORTED'] == 'true'\n"
        "    expected = os.environ['TEST_ARCHIVE_DIGEST'] if imported else os.environ['TEST_MANIFEST']\n"
        "    sys.exit(0 if present and sys.argv[3] == expected else 1)\n"
        "if sys.argv[1] == 'info':\n"
        "    print('name=rootless')\n"
    )
    docker.chmod(0o755)
    flock = commands / "flock"
    flock.write_text("#!/bin/sh\nexit 0\n")
    flock.chmod(0o755)
    action = yaml.safe_load(
        (ROOT / ".github/actions/offline-cpu/action.yml").read_text()
    )
    workspace = tmp_path / "workspace"
    lock = workspace / "docker/ci/images.lock"
    lock.parent.mkdir(parents=True)
    manifest = "ghcr.io/flagos-ai/flagquantum-ci@sha256:" + "1" * 64
    archive_digest = "sha256:" + "2" * 64
    lock.write_text(f"core-py3.12 {manifest} {archive_digest}\n")
    attach = next(
        step["run"]
        for step in action["runs"]["steps"]
        if "FQ_CI_PROFILE" in step.get("env", {})
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", attach],
        cwd=workspace,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TEST_DOCKER_LOG": str(log),
            "TEST_IMAGE_PRESENT": str(image_present).lower(),
            "TEST_IMPORTED": str(imported).lower(),
            "TEST_ARCHIVE_DIGEST": archive_digest,
            "TEST_MANIFEST": manifest,
            "FQ_CI_REQUIRE_CACHED": "true",
            "FQ_CI_HOST_BUDGET": "",
            "FQ_CI_PROFILE": "core-py3.12",
            "FQ_CI_WHEEL_DIRECTORY": "",
            "GITHUB_WORKSPACE": str(ROOT),
            "RUNNER_TEMP": str(tmp_path),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "RUNNER_NAME": "test-provisioned-cpu",
            "GITHUB_ENV": str(tmp_path / "environment"),
            "GITHUB_PATH": str(tmp_path / "path"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(call[0] not in {"login", "pull"} for call in calls)
    assert result.returncode == (0 if image_present else 1), result.stderr
    if image_present:
        expected = archive_digest if imported else manifest
        assert f"FQ_CI_IMAGE={expected}" in (tmp_path / "environment").read_text()
        run = next(call for call in calls if call[0] == "run")
        assert run[run.index("--network") + 1] == "none"
        assert run[run.index("--pull") + 1] == "never"
    else:
        assert "not provisioned" in result.stdout
        assert not any(call[0] == "run" for call in calls)


@pytest.mark.parametrize("fault", ["none", "changed_identity", "corrupt_manifest"])
def test_export_pins_oci_manifests_instead_of_local_docker_ids(
    tmp_path: Path, fault: str
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/prewarm-ci-cpu.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["export"]["steps"]
        if "python3 - <<'PYTHON'" in step.get("run", "")
    )
    script = command.split("python3 - <<'PYTHON'\n", 1)[1].split("\nPYTHON", 1)[0]
    files: dict[str, bytes] = {}
    descriptors = []
    rows = []
    for index, profile in enumerate(
        ("core-py3.10", "core-py3.11", "core-py3.12", "coverage-py3.12")
    ):
        reference = str(index + 1) * 64
        payload = json.dumps(
            {"schemaVersion": 2, "config": {"digest": reference}}
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()
        files[f"blobs/sha256/{digest}"] = (
            payload + b"corrupt"
            if fault == "corrupt_manifest" and index == 0
            else payload
        )
        descriptors.append(
            {
                "digest": f"sha256:{digest}",
                "annotations": {"org.opencontainers.image.ref.name": reference},
            }
        )
        expected = "0" * 64 if fault == "changed_identity" and index == 0 else digest
        rows.append(
            f"{profile} ghcr.io/flagos-ai/flagquantum-ci@sha256:{reference} sha256:{expected}\n"
        )
    files["index.json"] = json.dumps(
        {
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": descriptors,
        }
    ).encode()
    with tarfile.open(tmp_path / "images.tar.gz", "w:gz") as archive:
        for name, payload in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    (tmp_path / "source.lock").write_text("".join(rows))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if fault == "none":
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "images.lock").read_text() == "".join(rows)
    else:
        assert result.returncode != 0
        assert not (tmp_path / "images.lock").exists()
        assert (
            "Export identity changed" in result.stderr
            if fault == "changed_identity"
            else "Exported manifest does not match its digest" in result.stderr
        )
