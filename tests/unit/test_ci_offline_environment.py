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
            "COVERAGE_CORE": "sysmon",
            "FQ_COVERAGE_SHARDS": "2",
            "FQ_COVERAGE_INPUTS": str(temporary / "coverage-inputs"),
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
    assert "COVERAGE_CORE" in environment
    assert "FQ_COVERAGE_SHARDS" in environment
    assert "FQ_COVERAGE_INPUTS" in environment
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
@pytest.mark.parametrize("producer", ["export", "publish"])
def test_export_pins_oci_manifests_instead_of_local_docker_ids(
    tmp_path: Path, fault: str, producer: str
) -> None:
    workflow = yaml.safe_load(
        (
            ROOT
            / ".github/workflows"
            / (
                "prewarm-ci-cpu.yml"
                if producer == "export"
                else "publish-dev-container.yml"
            )
        ).read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"][
            "export" if producer == "export" else "publish-ci-cpu"
        ]["steps"]
        if "<<'PYTHON'" in step.get("run", "")
    )
    script = command.split("<<'PYTHON'\n", 1)[1].split("\nPYTHON", 1)[0]
    files: dict[str, bytes] = {}
    descriptors = []
    rows = []
    profiles = ("core-py3.10", "core-py3.11", "core-py3.12", "coverage-py3.12")
    for index, profile in enumerate(profiles if producer == "export" else profiles[:1]):
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
                "annotations": {
                    "org.opencontainers.image.ref.name": (
                        "0" * 64
                        if producer == "publish" and fault == "changed_identity"
                        else reference
                    )
                },
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
    published_archive = io.BytesIO()
    with tarfile.open(fileobj=published_archive, mode="w") as archive:
        for name, payload in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PROFILE": "core",
            "PYTHON_VERSION": "3.10",
            "FQ_CI_REGISTRY_IMAGE": "ghcr.io/flagos-ai/flagquantum-ci@sha256:"
            + "1" * 64,
        },
        input=published_archive.getvalue() if producer == "publish" else None,
        capture_output=True,
        timeout=10,
    )
    if fault == "none":
        assert result.returncode == 0, result.stderr
        actual = (
            (tmp_path / "images.lock").read_text()
            if producer == "export"
            else result.stdout.decode()
        )
        assert actual == "".join(rows)
    else:
        assert result.returncode != 0
        assert not (tmp_path / "images.lock").exists()
        assert (
            (
                "Export identity changed"
                if producer == "export"
                else "Exported reference does not match"
            )
            in result.stderr.decode()
            if fault == "changed_identity"
            else "Exported manifest does not match its digest" in result.stderr.decode()
        )


@pytest.mark.parametrize(
    ("edit", "expected_exit"),
    [
        ('license-files = ["LICENSE", "vendor/LICENSE"]', 0),
        ('target-version = "py312"', 0),
        ('build-backend = "other.backend"', 1),
        ('dependencies = ["torch>=2.14"]', 1),
        ('requires = ["setuptools>=78", "wheel"]', 1),
        ('qec = ["stim>=1.17"]', 1),
    ],
)
def test_offline_manifest_ignores_wheel_metadata_and_rejects_dependency_changes(
    tmp_path: Path, edit: str, expected_exit: int
) -> None:
    original = """\
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"
[project]
dependencies = ["torch>=2.13"]
license-files = ["LICENSE"]
[project.optional-dependencies]
qec = ["stim>=1.16"]
[tool.ruff]
target-version = "py310"
[tool.ruff.lint.per-file-ignores]
"flagquantum/observables/__init__.py" = ["E743"]
[tool.cibuildwheel]
test-command = "python smoke.py"
"""
    image = tmp_path / "image.toml"
    image.write_text(original)
    checkout = tmp_path / "checkout.toml"
    key, _ = edit.split(" = ", 1)
    updated = "\n".join(
        edit if line.startswith(f"{key} = ") else line for line in original.splitlines()
    )
    checkout.write_text(
        updated.replace("python smoke.py", "python check_wheel.py").replace(
            "flagquantum/observables/__init__.py",
            "flagquantum/core/operators/observable.py",
        )
        + "\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/actions/offline-cpu/check_manifest.py"),
            str(checkout),
            str(image),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_exit, result.stderr
    if expected_exit:
        assert "differs from the locked CPU image manifest" in result.stderr


@pytest.mark.parametrize(
    ("tables", "fault", "expected_exit"),
    [
        ("pytest", "none", 0),
        ("coverage", "none", 0),
        ("both", "none", 0),
        ("both", "runtime_dependency", 1),
        ("both", "optional_dependency", 1),
        ("both", "build_requirement", 1),
        ("both", "build_backend", 1),
        ("both", "python_constraint", 1),
        ("both", "unknown_tool", 1),
        ("both", "packaging_tool", 1),
    ],
)
def test_offline_manifest_accepts_checkout_test_config_but_keeps_install_identity(
    tmp_path: Path, tables: str, fault: str, expected_exit: int
) -> None:
    original = """\
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
[project]
name = "fixture"
requires-python = ">=3.10,<3.13"
dependencies = ["torch>=2.13"]
[project.optional-dependencies]
qec = ["stim>=1.16"]
"""
    pytest_config = """\
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = ["--strict-markers"]
markers = ["unit: local unit tests"]
"""
    coverage_config = """\
[tool.coverage.run]
source = ["flagquantum"]
relative_files = true
omit = ["flagquantum/benchmarking/*"]
[tool.coverage.report]
exclude_also = ["if TYPE_CHECKING:"]
"""
    image = tmp_path / "image.toml"
    image.write_text(original, encoding="utf-8")
    updated = original
    if tables in ("pytest", "both"):
        updated += pytest_config
    if tables in ("coverage", "both"):
        updated += coverage_config
    replacements = {
        "runtime_dependency": ("torch>=2.13", "torch>=2.14"),
        "optional_dependency": ("stim>=1.16", "stim>=1.17"),
        "build_requirement": ("setuptools>=77", "setuptools>=78"),
        "build_backend": ("setuptools.build_meta", "other.backend"),
        "python_constraint": (">=3.10,<3.13", ">=3.11,<3.13"),
    }
    if fault in replacements:
        old, new = replacements[fault]
        updated = updated.replace(old, new)
    elif fault == "unknown_tool":
        updated += "[tool.future_build]\nsetting = true\n"
    elif fault == "packaging_tool":
        updated += "[tool.setuptools]\ninclude-package-data = false\n"
    checkout = tmp_path / "checkout.toml"
    checkout.write_text(updated, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/actions/offline-cpu/check_manifest.py"),
            str(checkout),
            str(image),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_exit, result.stderr
    if expected_exit:
        assert "differs from the locked CPU image manifest" in result.stderr


def test_coverage_aggregation_uses_the_producers_locked_environment() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for name in ("coverage-plan", "coverage-shard", "coverage"):
        steps = jobs[name]["steps"]
        environment = [
            step
            for step in steps
            if step.get("uses") == "./.github/actions/offline-cpu"
        ]
        assert len(environment) == 1
        assert environment[0]["with"]["profile"] == "coverage-py3.12"
        assert environment[0]["with"]["require-cached"] == (
            "${{ runner.environment == 'self-hosted' }}"
        )
        assert not any("pip install" in step.get("run", "") for step in steps)
    aggregate = jobs["coverage"]
    assert aggregate["needs"] == ["prepare-source", "coverage-plan", "coverage-shard"]
    assert aggregate["timeout-minutes"] == 20
    commands = [step.get("run", "") for step in aggregate["steps"]]
    assert "test '${{ needs.coverage-shard.result }}' = success" in commands
    assert "python tools/ci/check_coverage.py" in commands


@pytest.mark.skipif(os.name == "nt", reason="Docker identity selection requires bash")
@pytest.mark.parametrize("security", ["rootless_tail", "rootless", "rootful", "failed"])
def test_docker_identity_consumes_complete_security_query_before_selection(
    tmp_path: Path, security: str
) -> None:
    """A producer closing after a positive match must not select the host UID."""
    commands = tmp_path / "commands"
    commands.mkdir()
    identity_command = commands / "id"
    identity_command.write_text(
        "#!" + sys.executable + "\n"
        "import sys\n"
        "print(1003 if sys.argv[1] == '-u' else 1004)\n"
    )
    identity_command.chmod(0o755)
    docker = commands / "docker"
    docker.write_text(
        "#!" + sys.executable + "\n"
        "import os, sys, time\n"
        "mode = os.environ['TEST_SECURITY']\n"
        "if mode == 'failed': sys.exit(23)\n"
        "if mode == 'rootful':\n"
        "    print('name=seccomp,profile=builtin')\n"
        "else:\n"
        "    os.write(1, b'name=rootless\\n')\n"
        "    if mode == 'rootless_tail':\n"
        "        time.sleep(0.1)\n"
        "        try: os.write(1, b'name=cgroupns\\n')\n"
        "        except BrokenPipeError: os._exit(141)\n"
    )
    docker.chmod(0o755)
    action = yaml.safe_load(
        (ROOT / ".github/actions/offline-cpu/action.yml").read_text()
    )
    attach = next(
        step["run"]
        for step in action["runs"]["steps"]
        if "FQ_CI_PROFILE" in step.get("env", {})
    )
    identity = attach.split("# Container UID 0", 1)[1]
    identity = identity[identity.index("container_user=") :]
    identity = identity.split('export FQ_CI_CONTAINER_USER="$container_user"', 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            identity + '\nprintf "%s" "$container_user"',
        ],
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TEST_SECURITY": security,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if security == "failed":
        assert result.returncode == 23
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        expected = "1003:1004" if security == "rootful" else "0:0"
        assert result.stdout == expected
