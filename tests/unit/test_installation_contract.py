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
    package = workflow["jobs"]["package"]
    steps = package["steps"]
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

    commands = "\n".join(_run_commands(package))
    assert "torch>=2.10,<2.11" in commands
    assert "torch>=2.13,<2.14" in commands


def test_release_build_reuses_the_validated_torch_environment() -> None:
    workflow = _workflow(".github/workflows/cd.yml")
    validate_commands = _run_commands(workflow["jobs"]["validate"])
    sdist_commands = _run_commands(workflow["jobs"]["build-sdist"])

    editable = next(command for command in validate_commands if " -e " in command)
    build = next(command for command in sdist_commands if "python -m build" in command)

    assert CPU_TORCH_INDEX in "\n".join(validate_commands + sdist_commands)
    assert "torch>=2.13,<2.14" in "\n".join(validate_commands)
    assert "torch>=2.10,<2.11" in "\n".join(sdist_commands)
    assert "--no-build-isolation" in editable
    assert "--no-isolation" in build


def test_release_wheel_jobs_provision_the_artifact_verifier() -> None:
    workflow = _workflow(".github/workflows/cd.yml")
    steps = workflow["jobs"]["build-wheels"]["steps"]

    setup_python = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/setup-python")
    ]
    assert len(setup_python) == 1
    assert setup_python[0]["with"]["python-version"] == "3.12"

    verify = next(step for step in steps if step.get("name") == "Verify wheel contents")
    assert verify["run"].startswith("python ")


def test_pull_request_ci_builds_native_release_wheels() -> None:
    workflow = _workflow(".github/workflows/ci.yml")
    job = workflow["jobs"]["release-wheel-platform"]
    matrix = job["strategy"]["matrix"]["include"]

    assert {
        (entry["os"], entry["artifact"], entry["identifier"]) for entry in matrix
    } == {
        ("macos-14", "macos-arm64", "cp310-macosx_arm64"),
        ("windows-2022", "windows-amd64", "cp310-win_amd64"),
    }
    assert any(
        str(step.get("uses", "")).startswith("pypa/cibuildwheel@")
        for step in job["steps"]
    )
    assert any(
        step.get("run") == "python tools/verify_distribution_artifacts.py wheelhouse"
        for step in job["steps"]
    )


def test_release_validation_lanes_are_parallel_and_dependency_complete() -> None:
    workflow = _workflow(".github/workflows/cd.yml")
    validate = workflow["jobs"]["validate"]
    lanes = validate["strategy"]["matrix"]["include"]

    assert lanes == [
        {"tier": "pr-default", "extras": "dev"},
        {"tier": "pr-runtime", "extras": "dev"},
        {"tier": "pr-distributed", "extras": "dev,jax"},
    ]
    assert validate["strategy"]["fail-fast"] is False
    assert validate["timeout-minutes"] >= 45

    for job_name in ("validate", "build-sdist", "build-wheels", "publish"):
        checkout = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout")
        )
        assert checkout["with"]["ref"] == "${{ env.RELEASE_REF }}"


def test_cibuildwheel_builds_one_abi3_wheel_against_the_stable_torch_abi() -> None:
    configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'build = "cp310-*"' in configuration
    assert "torch>=2.10,<2.11" in configuration
    assert "torch>=2.14,<2.15" in configuration
    assert "torch==2.5.*" not in configuration
    assert "manylinux_2_28" in configuration
    assert "delocate-wheel" in configuration
    assert "--ignore-missing-dependencies" in configuration
    assert "delvewheel repair" in configuration

    setup_configuration = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'extra_compile_args={"cxx": CPP_FLAGS}' in setup_configuration
    assert '"/std:c++17"' in setup_configuration
    assert "py_limited_api=True" in setup_configuration
    assert '"py_limited_api": "cp310"' in setup_configuration
    assert 'self.compiler.compiler_type == "msvc"' in setup_configuration
    assert "original_spawn = self.compiler.spawn" in setup_configuration
    assert 'executable in {"cl", "cl.exe"}' in setup_configuration
    assert "self.compiler.spawn = original_spawn" in setup_configuration
