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
    _forbidden,
    _is_native_extension,
    _missing_native_extension,
    _missing_required_members,
    artifact_errors,
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

# `python -m build` writes to `dist` unless `--outdir` says otherwise.
_BUILD_OUTPUT = re.compile(r"\bpython\s+-m\s+build\b(?:[^\n]*?--outdir[=\s]+(\S+))?")
# The verifier takes exactly one directory, which is the directory it inspects.
_ARTIFACT_VERIFICATION = re.compile(
    r"\btools/verify_distribution_artifacts\.py\s+(\S+)"
)
_PUBLISH_ACTION = "pypa/gh-action-pypi-publish"


def _fake_wheel(tmp_path: Path, *, native_extension: str | None) -> Path:
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
            "Requires-Dist: torch>=2.5,<2.14\n",
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
    assert not _forbidden("flagquantum/core/ir/__init__.py")


def test_distribution_allows_only_flagquantum_package_root() -> None:
    assert ALLOWED_WHEEL_PACKAGE_ROOTS == ("flagquantum/",)
    assert _allowed_wheel_member("flagquantum/core/ir/__init__.py")
    assert _allowed_wheel_member("flagquantum/ecosystem/qiskit/aer.py")
    assert _allowed_wheel_member("flagquantum-0.2.0.dist-info/METADATA")
    assert not _allowed_wheel_member("unrelated_plugin/__init__.py")


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
    """The published file must be the file this workflow inspected.

    `ci.yml`'s `package` job verifies the artifacts it builds and stops there,
    and `python -m build` in the release job is a second, independent build. So
    the file users receive was the only one no gate had opened: a build that
    dropped the compiled `_C` module, or that swept in tests and caches, would
    have been uploaded successfully and then described, elsewhere, as verified.
    The release must therefore run the same verifier over the directory it built.
    """

    steps = _publishing_job()["steps"]
    commands = [str(step.get("run", "")) for step in steps]

    built = [
        match.group(1) or "dist"
        for command in commands
        for match in [_BUILD_OUTPUT.search(command)]
        if match
    ]
    assert built, "the release builds the distributions it publishes"

    verified = [
        match.group(1)
        for command in commands
        for match in [_ARTIFACT_VERIFICATION.search(command)]
        if match
    ]
    assert sorted(verified) == sorted(built)

    publication = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith(_PUBLISH_ACTION)
    )
    verification = next(
        index
        for index, command in enumerate(commands)
        if _ARTIFACT_VERIFICATION.search(command)
    )
    assert verification < publication
