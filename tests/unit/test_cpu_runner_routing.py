"""Keep the first CPU migration on verified labels and off fork pull requests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
FIRST_BATCH = {
    "quality",
    "cpu-core",
    "cpu-core-3-12-tests",
    "cpu-runtime",
    "cpu-core-3-12",
    "distributed-cpu",
}
SHARED_ROUTE = (
    "${{ fromJSON((github.event_name == 'push' || "
    "(github.event_name == 'pull_request' && "
    "github.event.pull_request.head.repo.full_name == github.repository)) "
    '&& \'["self-hosted","Linux","X64","a100","jiuding","gpu"]\' '
    "|| '[\"ubuntu-latest\"]') }}"
)


@pytest.mark.parametrize("workflow", ["ci.yml", "pre-commit.yaml"])
def test_first_batch_uses_verified_labels_only_for_trusted_events(
    workflow: str,
) -> None:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    expected = FIRST_BATCH if workflow == "ci.yml" else {"pre-commit"}
    routed = {
        name
        for name, job in document["jobs"].items()
        if "self-hosted" in str(job.get("runs-on", ""))
    }
    assert routed == expected
    for name in routed:
        job = document["jobs"][name]
        # Explicit event allowlist: a missing PR payload cannot authorize a runner.
        assert job["runs-on"] == SHARED_ROUTE
        assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert job["timeout-minutes"] == 45
        for index, step in enumerate(job["steps"]):
            if str(step.get("uses", "")).startswith("actions/setup-python@"):
                assert job["steps"][index + 1]["uses"] == (
                    "./.github/actions/isolated-python"
                )
    triggers = document.get("on") or document.get(True, {})
    assert set(triggers) == {"push", "pull_request"}
    if workflow == "ci.yml":
        for name, job in document["jobs"].items():
            if name not in expected and name not in {
                "package",
                "release-wheel-platform",
            }:
                assert job["runs-on"] == "ubuntu-latest"


def test_isolation_rejects_host_packages_and_reuses_no_previous_environment(
    tmp_path: Path,
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/isolated-python/action.yml").read_text()
    )
    script = action["runs"]["steps"][0]["run"]
    host_packages = tmp_path / "host-packages"
    host_packages.mkdir()
    (host_packages / "fq_host_only.py").write_text("HOST_PACKAGE = True\n")
    base_env = os.environ.copy()
    base_env.update(
        RUNNER_TEMP=str(tmp_path),
        PYTHONPATH=str(host_packages),
        PYTHONHOME="/unavailable-host-python",
        FQ_BASE_ENV="/unverified-preinstalled-stack",
        ACTIONS_RUNNER_HOOK_JOB_STARTED="/unavailable-hook",
        PATH=str(Path(sys.executable).parent) + os.pathsep + base_env["PATH"],
    )
    environments = []
    for attempt in range(2):
        path_file = tmp_path / f"path-{attempt}"
        env_file = tmp_path / f"env-{attempt}"
        env = {**base_env, "GITHUB_PATH": str(path_file), "GITHUB_ENV": str(env_file)}
        subprocess.run(["bash", "-eu", "-c", script], env=env, check=True, timeout=45)
        exported = dict(
            line.split("=", 1) for line in env_file.read_text().splitlines()
        )
        venv = Path(exported["VIRTUAL_ENV"])
        environments.append(venv)
        assert path_file.read_text().strip() == str(venv / "bin")
        assert (
            "include-system-site-packages = false" in (venv / "pyvenv.cfg").read_text()
        )
        subprocess.run(
            [
                str(venv / "bin/python"),
                "-c",
                "import importlib.util, sys; "
                "assert sys.prefix != sys.base_prefix; "
                "assert importlib.util.find_spec('fq_host_only') is None",
            ],
            env={**env, **exported},
            check=True,
            timeout=10,
        )
    assert environments[0] != environments[1]
