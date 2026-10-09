from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def _workflow(path: str) -> dict[str, object]:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _run_commands(job: dict[str, object]) -> list[str]:
    return [
        str(step.get("run", ""))
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    ]


def test_readme_avoids_implicit_accelerator_resolution() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    cpu_torch = readme.index(CPU_TORCH_INDEX)
    cpu_install = readme.index("python -m pip install flagquantum", cpu_torch)

    assert cpu_torch < cpu_install
    assert "downloads about 200 MB" in readme
    assert "CONTRIBUTING.md" in readme


def test_contributor_install_reuses_the_selected_torch() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert CPU_TORCH_INDEX in contributing
    assert "python -m pip install --no-build-isolation -e '.[dev]'" in contributing


def test_ci_editable_installs_reuse_the_cpu_torch_environment() -> None:
    workflow = _workflow(".github/workflows/ci.yml")

    for name, job in workflow["jobs"].items():
        commands = _run_commands(job)
        editable = [command for command in commands if " -e " in command]
        if not editable:
            continue

        assert all("--no-build-isolation" in command for command in editable), name
        assert any(CPU_TORCH_INDEX in command for command in commands), name


def test_ci_measures_a_bounded_cold_cpu_install() -> None:
    workflow = _workflow(".github/workflows/ci.yml")
    steps = workflow["jobs"]["package"]["steps"]
    measured = [
        step
        for step in steps
        if isinstance(step, dict) and "Cold-install CPU wheel" in step.get("name", "")
    ]

    assert len(measured) == 1
    step = measured[0]
    assert step["timeout-minutes"] <= 5
    assert step["env"]["PIP_CACHE_DIR"].startswith("/tmp/")
    assert "GITHUB_STEP_SUMMARY" in step["run"]


def test_release_build_reuses_the_validated_torch_environment() -> None:
    workflow = _workflow(".github/workflows/cd.yml")
    commands = _run_commands(workflow["jobs"]["publish"])

    editable = next(command for command in commands if " -e " in command)
    build = next(command for command in commands if "python -m build" in command)

    assert CPU_TORCH_INDEX in "\n".join(commands)
    assert "--no-build-isolation" in editable
    assert "--no-isolation" in build
