import ast
import importlib.util
import re
import zipfile
from pathlib import Path

import pytest
import yaml

from tools.artifact_manifest import main as manifest_main
from tools.verify_distribution_artifacts import (
    ALLOWED_WHEEL_PACKAGE_ROOTS,
    REQUIRED_MEMBER_SUFFIXES,
    _allowed_wheel_member,
    _bundled_torch_library,
    _forbidden,
    _is_native_extension,
    _missing_native_extension,
    _missing_required_members,
    artifact_errors,
    release_matrix_errors,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

DIST_INFO = "flagquantum-0.2.0.dist-info"
NATIVE_EXTENSION = (
    "flagquantum/simulation/native_cpu/_C.cpython-312-x86_64-linux-gnu.so"
)

RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "cd.yml"

_PUBLISH_ACTION = "pypa/gh-action-pypi-publish"


def _fake_wheel(
    tmp_path: Path,
    *,
    native_extension: str | None,
    torch_requirement: str = "torch>=2.13,<2.14",
) -> Path:
    """A minimal wheel that satisfies every artifact rule except the extension.

    The checks below are about what the verifier requires of a real build, so
    the fixture is a real archive that the verifier opens rather than a list of
    member names handed to one helper.
    """

    members = [
        "flagquantum/__init__.py",
        "flagquantum/simulation/native_cpu/adjoint.py",
        f"{DIST_INFO}/licenses/LICENSE",
        *REQUIRED_MEMBER_SUFFIXES,
    ]
    if native_extension is not None:
        members.append(native_extension)
    path = tmp_path / "flagquantum-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")
        archive.writestr(
            f"{DIST_INFO}/METADATA",
            "Metadata-Version: 2.1\nName: flagquantum\n"
            f"Requires-Dist: {torch_requirement}\n"
            'Requires-Dist: tomli>=2.0,<3; python_version < "3.11"\n',
        )
    return path


def test_package_uses_one_dynamic_version_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "flagquantum.version.__version__"}' in pyproject
    assert '\nversion = "0.1.0"' not in pyproject


def test_ci_python_paths_resolve():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    paths = set(re.findall(r"\bflagquantum/[A-Za-z0-9_./-]+", workflow))
    assert paths
    assert not [path for path in paths if not (ROOT / path).exists()]


def test_ci_install_checks_import_existing_modules():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    snippets = re.findall(r'python -c "([^"\n]+)"', workflow)
    modules = {
        node.module
        for snippet in snippets
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("flagquantum.")
    }
    assert modules
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module


def test_clean_install_checks_do_not_import_the_checkout() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    wheel_check = workflow.index("/tmp/fq-wheel/bin/python -c")
    sdist_check = workflow.index("/tmp/fq-sdist/bin/python -c")
    assert workflow.rfind("pushd /tmp", 0, wheel_check) != -1
    assert workflow.rfind("pushd /tmp", 0, sdist_check) > wheel_check


def test_distribution_quarantine_rejects_repo_only_members():
    assert _forbidden("flagquantum/__pycache__/module.pyc")
    assert _forbidden("source/tests/test_api.py")
    assert _forbidden("source/examples/data.parquet")
    assert not _forbidden("flagquantum/core/ir.py")


def test_distribution_allows_only_flagquantum_package_root() -> None:
    assert ALLOWED_WHEEL_PACKAGE_ROOTS == ("flagquantum/", "flagquantum.libs/")
    assert _allowed_wheel_member("flagquantum/core/ir.py")
    assert _allowed_wheel_member("flagquantum/ecosystem/qiskit/aer.py")
    assert _allowed_wheel_member("flagquantum.libs/libgomp.so.1")
    assert _allowed_wheel_member("flagquantum-0.2.0.dist-info/METADATA")
    assert not _allowed_wheel_member("unrelated_plugin/__init__.py")


def test_distribution_keeps_pytorch_runtime_libraries_external() -> None:
    assert _bundled_torch_library("flagquantum.libs/libtorch_cpu.so")
    assert _bundled_torch_library("flagquantum.libs/c10.dll")
    assert _bundled_torch_library("flagquantum/.dylibs/libc10.dylib")
    assert not _bundled_torch_library("flagquantum.libs/libgomp.so.1")


def test_distribution_rejects_an_abi_incompatible_torch_requirement(
    tmp_path: Path,
) -> None:
    wheel = _fake_wheel(
        tmp_path,
        native_extension=NATIVE_EXTENSION,
        torch_requirement="torch>=2.5,<2.14",
    )

    assert any("PyTorch 2.13 ABI" in error for error in artifact_errors(wheel))


def test_distribution_requires_runtime_profiles_and_numerical_contract() -> None:
    assert _missing_required_members(REQUIRED_MEMBER_SUFFIXES) == ()
    assert _missing_required_members(REQUIRED_MEMBER_SUFFIXES[1:]) == (
        "flagquantum/simulation/numerics/double-single-contract.toml",
    )


def test_distribution_requires_the_compiled_native_extension(tmp_path) -> None:
    """`setup.py` builds `_C`, so a wheel without it is an incomplete artifact.

    The release path asserts `native_cpu_adjoint_available()` in a separate
    job, and the fix here is that this gate stops the wheel before a gate that
    lives elsewhere has to notice.
    """

    assert (
        artifact_errors(_fake_wheel(tmp_path, native_extension=NATIVE_EXTENSION)) == ()
    )
    errors = artifact_errors(_fake_wheel(tmp_path, native_extension=None))

    assert not _missing_native_extension(("x.so", NATIVE_EXTENSION))
    assert (
        "compiled native extension is missing: flagquantum.simulation.native_cpu._C"
        in errors
    )


def test_native_extension_members_are_matched_by_directory_and_module_name() -> None:
    assert _is_native_extension(NATIVE_EXTENSION)
    assert _is_native_extension(
        "flagquantum/simulation/native_cpu/_C.cp310-win_amd64.pyd"
    )
    assert _is_native_extension("flagquantum/simulation/native_cpu/_C.so")
    assert not _is_native_extension("flagquantum/simulation/native_cpu/_D.so")
    assert not _is_native_extension("flagquantum/simulation/native_cpu/adjoint.py")
    assert not _is_native_extension(
        "flagquantum/simulation/native_cpu/csrc/permutation.cpp"
    )
    assert not _is_native_extension("flagquantum/other/_C.cpython-312-darwin.so")


def test_release_matrix_requires_every_supported_wheel_and_one_sdist(
    tmp_path: Path,
) -> None:
    platforms = (
        "manylinux_2_28_x86_64",
        "macosx_11_0_arm64",
        "win_amd64",
    )
    artifacts = tuple(
        tmp_path / f"flagquantum-0.2.0-{python}-{python}-{platform}.whl"
        for python in ("cp310", "cp311", "cp312")
        for platform in platforms
    ) + (tmp_path / "flagquantum-0.2.0.tar.gz",)

    assert release_matrix_errors(artifacts) == ()
    errors = release_matrix_errors(artifacts[1:])
    assert "release wheel is missing: cp310 manylinux_x86_64" in errors


def test_project_description_matches_the_readme_positioning() -> None:
    """PyPI shows this line, so it must not drift away from the README."""

    description = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["description"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert description == (
        "A PyTorch-first framework for differentiable quantum computing "
        "and quantum AI."
    )
    assert description in readme


def test_artifact_manifest_records_checksum(tmp_path):
    artifact = tmp_path / "sample.whl"
    artifact.write_bytes(b"flagquantum")
    output = tmp_path / "manifest.json"

    assert manifest_main(["--output", str(output), str(artifact)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "flagquantum_artifact_manifest_v1" in text
    assert "sample.whl" in text
    assert "sha256" in text


def _publishing_job() -> dict[str, object]:
    """The one job that uploads to PyPI."""

    document = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    jobs = document["jobs"]
    publishing = [
        job
        for job in jobs.values()
        if any(
            isinstance(step, dict)
            and str(step.get("uses", "")).startswith(_PUBLISH_ACTION)
            for step in job["steps"]
        )
    ]
    assert len(publishing) == 1, "exactly one job may publish to PyPI"
    return publishing[0]


def test_the_release_verifies_the_artifact_it_publishes() -> None:
    """The publisher downloads and verifies the complete matrix before upload."""

    job = _publishing_job()
    steps = job["steps"]
    commands = [str(step.get("run", "")) for step in steps]
    assert set(job["needs"]) == {"build-sdist", "build-wheels"}

    publication = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith(_PUBLISH_ACTION)
    )
    download = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/download-artifact")
    )
    verification = next(
        index
        for index, command in enumerate(commands)
        if "verify_distribution_artifacts.py --release-matrix dist" in command
    )
    assert download < verification < publication


def test_release_builds_the_supported_platform_and_interpreter_matrix() -> None:
    document = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    job = document["jobs"]["build-wheels"]
    platforms = {
        (entry["os"], entry["artifact"])
        for entry in job["strategy"]["matrix"]["include"]
    }

    assert platforms == {
        ("ubuntu-22.04", "manylinux-x86_64"),
        ("macos-14", "macos-arm64"),
        ("windows-2022", "windows-amd64"),
    }
    assert job["strategy"]["max-parallel"] <= 4
    assert any(
        str(step.get("uses", "")).startswith("pypa/cibuildwheel@")
        for step in job["steps"]
    )
