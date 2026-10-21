import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import setuptools
import torch
import yaml
from setuptools.dist import Distribution
from torch.utils.cpp_extension import BuildExtension

from tools.check_dependency_policy import load_toml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def _workflow(path: str) -> dict[str, object]:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _run_commands(job: dict[str, object]) -> list[str]:
    commands = []
    for step in job["steps"]:
        if not isinstance(step, dict):
            continue
        if "run" in step:
            commands.append(str(step["run"]))
        elif step.get("uses") == "./.github/actions/install-cpu-torch":
            action = _workflow(".github/actions/install-cpu-torch/action.yml")
            commands.extend(_run_commands(action["runs"]))
    return commands


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
        cpu_install = next(
            index
            for index, command in enumerate(commands)
            if CPU_TORCH_INDEX in command
        )
        project_install = next(
            index for index, command in enumerate(commands) if " -e " in command
        )
        assert cpu_install < project_install, name


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
    assert '["/openmp:experimental"]' in setup_configuration
    assert "py_limited_api=True" in setup_configuration
    assert '"py_limited_api": "cp310"' in setup_configuration
    assert 'os.environ.get("CL", "")' in setup_configuration
    assert 'os.environ["CL"] =' in setup_configuration
    assert "_configure_msvc_environment()" in setup_configuration


def test_one_wheel_is_loaded_on_the_older_runtime_minors() -> None:
    workflow = _workflow(".github/workflows/ci.yml")
    step = next(
        item
        for item in workflow["jobs"]["package"]["steps"]
        if item.get("name") == "Verify one wheel on older PyTorch runtimes"
    )
    command = step["run"]
    assert "for version in 2.10 2.11 2.12" in command
    assert "pip install dist/*.whl" in command
    assert "native_cpu_adjoint_available()" in command
    assert "cd /tmp" in command
    assert step["timeout-minutes"] >= 10


def test_build_backend_supports_spdx_license_metadata() -> None:
    project = load_toml(ROOT / "pyproject.toml")
    assert project["project"]["license"] == "Apache-2.0"
    assert "setuptools>=77.0.3" in project["build-system"]["requires"]


def test_native_extension_avoids_compiler_specific_bit_builtins() -> None:
    source = (
        ROOT / "flagquantum/simulation/native_cpu/csrc/linear_permutation.h"
    ).read_text(encoding="utf-8")

    assert "__builtin_" not in source
    assert "trailing_zero_count" in source


def _build_ext_command(inplace: bool):
    """Return a configured `build_ext` command without running a build.

    `setup.py` cannot be imported, so it is executed with `setuptools.setup`
    replaced by a recorder. That yields the real command class and the real
    extension declarations, which is what the linker arguments are computed
    from.
    """

    captured: dict[str, object] = {}
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "fq_setup_probe"}
    with mock.patch.object(
        setuptools, "setup", lambda **kwargs: captured.update(kwargs)
    ):
        exec(compile(source, "setup.py", "exec"), namespace)
    command = captured["cmdclass"]["build_ext"](
        Distribution({"ext_modules": captured["ext_modules"]})
    )
    command.initialize_options()
    command.inplace = inplace
    command.finalize_options()
    return command


def _torch_library_rpath() -> str:
    return f"-Wl,-rpath,{Path(torch.__file__).parent / 'lib'}"


def _relocatable_torch_rpath() -> str:
    origin = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    return f"-Wl,-rpath,{origin}/../../../torch/lib"


def test_an_inplace_build_adds_this_machines_torch_library_to_the_rpath() -> None:
    """An in-place artifact sits where the relocatable rpath cannot resolve.

    The compiled extension lands in `<source>/flagquantum/simulation/native_cpu/`
    and carries a path relative to itself, which from there is
    `<source>/torch/lib` -- absent, so `dlopen` fails and
    `adjoint._load_extension` reports the native operator as unavailable.
    """

    if os.name == "nt":
        pytest.skip("Windows resolves the PyTorch DLLs through PATH")

    link_args = _build_ext_command(inplace=True).extensions[0].extra_link_args

    assert _relocatable_torch_rpath() in link_args
    assert link_args.count(_torch_library_rpath()) == 1


def test_a_distributed_build_keeps_only_the_relocatable_rpath() -> None:
    """A wheel must not carry the build machine's PyTorch path.

    `tools/verify_reproducible_builds.py` compares two isolated builds byte for
    byte, so an absolute path recorded at link time would differ between them.
    """

    link_args = _build_ext_command(inplace=False).extensions[0].extra_link_args

    assert _torch_library_rpath() not in link_args


def test_the_load_check_follows_the_inplace_flag(monkeypatch) -> None:
    """The delivered artifact is opened only after an in-place build.

    setuptools clears `inplace` while the extensions are built and copies them
    into the source tree afterwards, so the check belongs to `run`, where the
    path it opens is the one the runtime loads.
    """

    checked: list[str] = []
    monkeypatch.setattr(BuildExtension, "run", lambda self: None)
    for inplace, expected in ((True, 1), (False, 0)):
        command = _build_ext_command(inplace=inplace)
        monkeypatch.setattr(
            command,
            "_require_loadable_artifact",
            lambda ext, name=f"{expected}": checked.append(name),
        )
        command.run()

    assert checked == ["1"]


def test_an_inplace_build_relinks_an_artifact_linked_by_an_earlier_build(
    monkeypatch, tmp_path
) -> None:
    """A staged artifact newer than its sources is reused, not relinked.

    The up-to-date test compares the sources against the staged artifact and
    never sees the link command, so an artifact left by an earlier build keeps
    that build's rpath. Removing the linked output forces this build's link
    arguments onto it and leaves the object files to be reused.
    """

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(BuildExtension, "build_extension", lambda self, ext: None)
    artifact = tmp_path / "_C.abi3.so"

    def does_the_artifact_survive(inplace: bool) -> bool:
        artifact.write_bytes(b"linked by an earlier build")
        command = _build_ext_command(inplace=inplace)
        command.get_ext_fullpath = lambda name: str(artifact)
        command.build_extension(command.extensions[0])
        return artifact.exists()

    assert does_the_artifact_survive(inplace=True) is False
    assert does_the_artifact_survive(inplace=False) is True


def test_an_inplace_build_fails_when_the_linked_artifact_cannot_load(
    tmp_path,
) -> None:
    """A build that reports success while `dlopen` fails is the defect.

    `_load_extension` turns that failure into `False` so an unbuilt extension
    degrades instead of raising; a built one that cannot load must not be
    reported as a successful build.
    """

    artifact = tmp_path / "_C.abi3.so"
    artifact.write_bytes(b"not a shared library")
    command = _build_ext_command(inplace=True)
    command.get_ext_fullpath = lambda name: str(artifact)

    with pytest.raises(RuntimeError, match="was built but cannot be loaded"):
        command._require_loadable_artifact(command.extensions[0])
