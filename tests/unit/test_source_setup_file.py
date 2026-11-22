"""Complete frozen setup entry and existing native metadata build consumer."""

import runpy
from collections.abc import Callable, Mapping
from email.parser import Parser
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from setuptools import find_packages
from setuptools.build_meta import prepare_metadata_for_build_wheel

from flagquantum.version import __version__

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
SOURCE_BLOBS = {
    "setup.py": "4eb99f1178be2d037f10e2716c2d564a1e04b6ce",
    "qualtran/_version.py": "a21f8f56cc9e68b4e36f36665fdeba877ced3630",
    "README.md": "74ce34b1ad2c8a5785b9b0ea5886a5a1b1076288",
    "dev_tools/requirements/deps/runtime.txt": (
        "fa08f464886079ffbc362afb0f1207c912c0a7c6"
    ),
}
SourceSetup = tuple[Path, Mapping[str, object], Mapping[str, object]]


@pytest.fixture
def source_setup(monkeypatch: pytest.MonkeyPatch) -> SourceSetup:
    """Execute the actual frozen script, intercepting distribution construction."""
    package = import_module("qualtran")
    assert package.__file__ is not None
    root = Path(package.__file__).resolve().parents[1]
    for name, blob in SOURCE_BLOBS.items():
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    monkeypatch.chdir(root)
    with patch("setuptools.setup") as setup:
        namespace = runpy.run_path(str(root / "setup.py"))
    setup.assert_called_once()
    return root, namespace, setup.call_args.kwargs


@pytest.mark.ftqc_reference
def test_complete_source_setup_metadata(source_setup: SourceSetup) -> None:
    root, namespace, observed = source_setup
    assert {name for name in namespace if not name.startswith("__")} == {
        "re",
        "find_packages",
        "setup",
        "version_number",
        "main",
    }
    assert observed == {
        "name": "qualtran",
        "version": "0.7.0",
        "author": "Google Quantum AI",
        "author_email": "mpharrigan@google.com",
        "description": "Software for fault-tolerant quantum algorithms research.",
        "long_description": (root / "README.md").read_text(),
        "install_requires": [
            line.strip()
            for line in (root / "dev_tools/requirements/deps/runtime.txt")
            .read_text()
            .splitlines()
            if not line.startswith("#")
        ],
        "license": "Apache 2",
        "packages": find_packages(where=str(root)),
    }
    packages = cast(list[str], observed["packages"])
    assert len(packages) == 69
    assert all(name == "qualtran" or name.startswith("qualtran.") for name in packages)
    assert "" in cast(list[str], observed["install_requires"])


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "value, expected",
    [
        (value, value)
        for value in ("0.7.0", "1.2.3", "0.7.0.dev1", "0.7.0.dev", "0.7.0d")
    ]
    + [("0.7.0a1", None), ("0.7.0rc1", None), ("invalid", None)],
)
def test_original_version_parser_boundaries(
    source_setup: SourceSetup, tmp_path: Path, value: str, expected: str | None
) -> None:
    _, namespace, _ = source_setup
    version_number = cast(Callable[[str], str], namespace["version_number"])
    path = tmp_path / "version.py"
    path.write_text(f'__version__ = "{value}"\n')
    if expected is None:
        with pytest.raises(AttributeError, match="group"):
            version_number(str(path))
    else:
        assert version_number(str(path)) == expected


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "missing",
    ("qualtran/_version.py", "README.md", "dev_tools/requirements/deps/runtime.txt"),
)
def test_source_setup_missing_inputs_fail_before_build(
    source_setup: SourceSetup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    root, _, _ = source_setup
    for name in SOURCE_BLOBS:
        if name in {"setup.py", missing}:
            continue
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / name).read_bytes())
    monkeypatch.chdir(tmp_path)
    with patch("setuptools.setup") as setup, pytest.raises(FileNotFoundError):
        runpy.run_path(str(root / "setup.py"))
    setup.assert_not_called()


def test_native_pep517_metadata_uses_existing_package_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real setup script through its metadata hook in temporary output."""
    monkeypatch.chdir(ROOT)
    name = prepare_metadata_for_build_wheel(str(tmp_path))
    metadata = Parser().parsestr((tmp_path / name / "METADATA").read_text())
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = configuration["project"]
    assert metadata["Name"] == project["name"] == "flagquantum"
    assert metadata["Version"] == __version__
    assert metadata["Summary"] == project["description"]
    assert metadata["License-Expression"] == project["license"]
    assert metadata["Author-email"] == "FlagQuantum Team <flagquantum@gmail.com>"
    assert metadata.get_payload() == (ROOT / project["readme"]).read_text()
    requirements = metadata.get_all("Requires-Dist", [])
    assert "torch<2.15,>=2.10" in requirements
    assert 'tomli<3,>=2.0; python_version < "3.11"' in requirements
    package_config = configuration["tool"]["setuptools"]["packages"]["find"]
    packages = find_packages(
        where=str(ROOT),
        include=package_config["include"],
        exclude=package_config["exclude"],
    )
    assert "flagquantum" in packages
    assert all(name.startswith("flagquantum") for name in packages)
