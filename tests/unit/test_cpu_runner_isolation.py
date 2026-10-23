"""Exercise private CPU environments, package mirrors and launcher isolation."""

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
WORKFLOWS = ROOT / ".github/workflows"


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
    runtime_library_path = os.pathsep.join(
        [
            entry
            for entry in base_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if entry
            and "/site-packages/" not in entry
            and "/dist-packages/" not in entry
        ]
        + ["/opt/selected-python/lib", "/usr/local/cuda/compat/lib"]
    )
    base_env.update(
        RUNNER_TEMP=str(tmp_path),
        PYTHONPATH=str(host_packages),
        PYTHONHOME="/unavailable-host-python",
        LD_LIBRARY_PATH=(
            "/usr/local/lib/python3.12/dist-packages/torch/lib:"
            "/opt/venv/lib/python3.12/site-packages/torch_tensorrt/lib:"
            + runtime_library_path
            + ":"
        ),
        FQ_BASE_ENV="/unverified-preinstalled-stack",
        ACTIONS_RUNNER_HOOK_JOB_STARTED="/unavailable-hook",
        PATH=str(Path(sys.executable).parent) + os.pathsep + base_env["PATH"],
    )
    environments = []
    for attempt in range(2):
        path_file = tmp_path / f"path-{attempt}"
        env_file = tmp_path / f"env-{attempt}"
        runner_environment = "self-hosted" if attempt == 0 else "github-hosted"
        env = {
            **base_env,
            "GITHUB_PATH": str(path_file),
            "GITHUB_ENV": str(env_file),
            "RUNNER_ENVIRONMENT": runner_environment,
        }
        subprocess.run(["bash", "-eu", "-c", script], env=env, check=True, timeout=45)
        exported = dict(
            line.split("=", 1) for line in env_file.read_text().splitlines()
        )
        if runner_environment == "self-hosted":
            assert (
                exported["PIP_INDEX_URL"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        else:
            assert exported["PIP_INDEX_URL"] == "https://pypi.org/simple"
        assert exported["LD_LIBRARY_PATH"] == runtime_library_path
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


def test_source_hygiene_checkout_does_not_require_unneeded_git_lfs() -> None:
    document = yaml.safe_load((WORKFLOWS / "pre-commit.yaml").read_text())
    checkouts = [
        step
        for step in document["jobs"]["pre-commit"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts
    assert "filter=lfs" not in (ROOT / ".gitattributes").read_text()
    assert all(not step.get("with", {}).get("lfs", False) for step in checkouts)


@pytest.mark.parametrize("cache_state", ["absent", "hit", "miss"])
@pytest.mark.parametrize("torch_spec", ["torch>=2.13,<2.14", "torch>=2.10,<2.11"])
@pytest.mark.parametrize(
    ("runner_environment", "mirror_failure"),
    [
        ("self-hosted", "none"),
        ("self-hosted", "error"),
        ("self-hosted", "timeout"),
        ("github-hosted", "none"),
    ],
)
def test_cpu_wheel_mirror_fallback_preserves_the_official_hosted_route(
    tmp_path: Path,
    runner_environment: str,
    mirror_failure: str,
    torch_spec: str,
    cache_state: str,
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/install-cpu-torch/action.yml").read_text()
    )
    cache = tmp_path / "cpu-wheels"
    if cache_state != "absent":
        cache.mkdir()
    log = tmp_path / "pip-calls.jsonl"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "with open(os.environ['PIP_CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == '-c' and 'Requirement' in sys.argv[2]:\n"
        "    raise SystemExit(1)\n"
        "if '--find-links' in sys.argv and os.environ['FQ_CPU_WHEEL_CACHE'] in sys.argv:\n"
        "    raise SystemExit(0 if os.environ['CACHE_STATE'] == 'hit' else 1)\n"
        "if 'download' in sys.argv:\n"
        "    if os.environ['MIRROR_FAILURE'] == 'error':\n"
        "        raise SystemExit(1)\n"
        "    destination = pathlib.Path(sys.argv[sys.argv.index('--dest') + 1])\n"
        "    (destination / 'torch-2.13.0+cpu.whl').touch()\n"
    )
    fake_python.chmod(0o755)
    fake_timeout = tmp_path / "timeout"
    fake_timeout.write_text(
        "#!/bin/bash\n"
        'test "$1" = --kill-after=10s && test "$2" = 180s || exit 2\n'
        'if [ "$MIRROR_FAILURE" = timeout ]; then exit 124; fi\n'
        "shift 2\n"
        'exec "$@"\n'
    )
    fake_timeout.chmod(0o755)
    subprocess.run(
        ["bash", "-eu", "-c", action["runs"]["steps"][0]["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PIP_CALL_LOG": str(log),
            "MIRROR_FAILURE": mirror_failure,
            "CACHE_STATE": cache_state,
            "FQ_CPU_WHEEL_CACHE": str(cache),
            "FQ_TORCH_SPEC": torch_spec,
            "RUNNER_ENVIRONMENT": runner_environment,
            "RUNNER_TEMP": str(tmp_path),
        },
        check=True,
        timeout=15,
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    pip_calls = [command for command in commands if command[:2] == ["-m", "pip"]]
    cached = [call for call in pip_calls if str(cache) in call]
    if runner_environment == "self-hosted" and cache_state != "absent":
        assert cached == [
            [
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--find-links",
                str(cache),
                torch_spec,
            ]
        ]
        if cache_state == "hit":
            assert len(pip_calls) == 2
            assert pip_calls[-1] == ["-m", "pip", "install", torch_spec]
            assert not list(tmp_path.glob("fq-cpu-wheel.*"))
            return
        pip_calls = [call for call in pip_calls if str(cache) not in call]
    else:
        assert not cached
    official = "https://download.pytorch.org/whl/cpu"
    mirror = "https://mirrors.aliyun.com/pytorch-wheels/cpu/"
    if runner_environment == "self-hosted" and mirror_failure != "timeout":
        assert mirror in pip_calls[0]
        assert "--no-index" in pip_calls[0] and "--no-deps" in pip_calls[0]
        assert pip_calls[0][2] == "download"
    if runner_environment == "self-hosted" and mirror_failure == "none":
        assert len(pip_calls) == 3
        assert official not in str(pip_calls)
        assert "--no-deps" in pip_calls[1]
        assert pip_calls[2] == ["-m", "pip", "install", torch_spec]
    else:
        assert official in pip_calls[-2]
        assert "--no-deps" in pip_calls[-2]
        assert torch_spec in pip_calls[-2]
    if runner_environment == "github-hosted":
        assert len(pip_calls) == 2
        assert mirror not in str(pip_calls)
    assert pip_calls[-1] == ["-m", "pip", "install", torch_spec]
    assert "--index-url" not in pip_calls[-1]
    assert commands[-1][0] == "-c"
    assert "torch.version.cuda is None" in commands[-1][1]
    assert not list(tmp_path.glob("fq-cpu-wheel.*"))


def test_shared_tier_clears_platform_ranks_without_changing_test_selection(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    command = next(
        step["run"]
        for step in workflow["jobs"]["distributed-cpu"]["steps"]
        if "python tools/ci_tier.py pr-distributed" in str(step.get("run", ""))
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    output = tmp_path / "result.json"
    interpreter = bindir / "python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FQ_RANK_TEST_RESULT'], 'w') as stream:\n"
        "    json.dump({'args': sys.argv[1:], 'env': dict(os.environ)}, stream)\n"
    )
    interpreter.chmod(0o755)
    inherited = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "GROUP_RANK": "0",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    environment = {
        **os.environ,
        **inherited,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "FQ_RANK_TEST_RESULT": str(output),
    }
    subprocess.run(["bash", "-ec", command], env=environment, check=True, timeout=20)
    result = json.loads(output.read_text())
    assert result["args"] == ["tools/ci_tier.py", "pr-distributed"]
    assert not (set(inherited) & set(result["env"]))
    assert environment["RANK"] == "0"
    # Launched coverage still runs under two real ranks on the A800 runtime job.
    runtime = workflow["jobs"]["cpu-runtime"]["steps"]
    assert any(
        step.get("run")
        == "torchrun --standalone --nproc-per-node=2 -m pytest -m distributed_launch -q"
        for step in runtime
    )


def test_jax_launcher_coverage_uses_actual_two_rank_topology() -> None:
    document = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    steps = document["jobs"]["distributed-cpu"]["steps"]
    launched = next(
        step
        for step in steps
        if "test_hybrid_jax_runtime.py" in str(step.get("run", ""))
    )
    assert launched["run"] == (
        "torchrun --standalone --nproc-per-node=2 -m pytest "
        "tests/distributed/test_hybrid_jax_runtime.py -q"
    )
    assert launched["timeout-minutes"] == 10
    assert "PYTEST_ADDOPTS" not in launched.get("env", {})


@pytest.mark.parametrize("job", ["cpu-core", "cpu-core-3-12-tests", "cpu-runtime"])
def test_package_indexes_match_routing_before_runner_assignment(job: str) -> None:
    document = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    # Job env is evaluated before runner assignment; runner context is unavailable.
    trusted = (
        "(github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository)"
    )
    selected = document["jobs"][job]
    assert selected["env"]["PIP_INDEX_URL"] == (
        "${{ " + trusted + " && "
        "'https://pypi.tuna.tsinghua.edu.cn/simple' || 'https://pypi.org/simple' }}"
    )
    # Manual validation is trusted too; fork PRs retain the hosted fallback.
    assert selected["runs-on"] == (
        "${{ fromJSON(" + trusted + " && "
        '\'["self-hosted","linux","x64","flagquantum-a800-cpu"]\' || '
        "'[\"ubuntu-latest\"]') }}"
    )
    assert trusted in selected["concurrency"]["group"]
    assert "github.run_id" in selected["concurrency"]["group"]


@pytest.mark.parametrize(
    ("version", "cuda", "ready"),
    [
        ("2.13.0+cpu", None, True),
        ("2.14.0+cpu", None, False),
        ("2.13.0+cu130", "13.0", False),
        ("2.13.0", None, False),
    ],
)
def test_cached_torch_probe_rejects_wrong_version_or_accelerator_build(
    tmp_path: Path, version: str, cuda: str | None, ready: bool
) -> None:
    import shlex

    action = yaml.safe_load(
        (ROOT / ".github/actions/install-cpu-torch/action.yml").read_text()
    )
    command = next(
        line.strip()
        for line in action["runs"]["steps"][0]["run"].splitlines()
        if line.strip().startswith("if python -c")
    )
    probe = shlex.split(command)[3]
    (tmp_path / "torch.py").write_text(
        f"__version__ = {version!r}\n" f"class version:\n    cuda = {cuda!r}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "FQ_TORCH_SPEC": "torch>=2.13,<2.14",
        },
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) == ready, result.stderr.decode()
