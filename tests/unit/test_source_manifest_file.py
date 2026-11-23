"""Complete source manifest and actual native source-distribution consumer."""

import tarfile
from email.parser import Parser
from hashlib import sha1
from pathlib import Path

import pytest
from setuptools._distutils.filelist import FileList
from setuptools.build_meta import build_sdist

from flagquantum.version import __version__

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"


@pytest.mark.ftqc_reference
def test_complete_source_manifest_dependency_inclusion() -> None:
    raw = (REFERENCE / "MANIFEST.in").read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "8f97b4d84b80065568b51b21fb6472250fc49da2"
    )
    files = FileList()
    files.set_allfiles(
        [
            "dev_tools/requirements/deps/runtime.txt",
            "dev_tools/requirements/deps/docs.txt",
            "unrelated.txt",
        ]
    )
    for line in raw.decode().splitlines():
        files.process_template_line(line)
    assert files.files == ["dev_tools/requirements/deps/runtime.txt"]


def test_native_sdist_retains_authoritative_build_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read the actual native sdist produced by the existing setuptools entry."""
    monkeypatch.chdir(ROOT)
    name = build_sdist(
        str(tmp_path),
        config_settings={"--global-option": ["egg_info", "--egg-base", str(tmp_path)]},
    )
    with tarfile.open(tmp_path / name, "r:gz") as archive:
        prefix = name.removesuffix(".tar.gz") + "/"
        members = {
            name.removeprefix(prefix)
            for name in archive.getnames()
            if name.startswith(prefix)
        }
        required = {
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "flagquantum/version.py",
            "flagquantum/__init__.py",
            "flagquantum/compiler/artifacts/_unsigned_size.py",
            "flagquantum/compiler/formats/kmx_builder.py",
            "flagquantum/compiler/registers/layout/wire_allocation.py",
            "flagquantum/compiler/registers/lookup/lookup_binding.py",
            "flagquantum/ecosystem/_physical_basis_session.py",
        }
        assert required <= members
        assert any(
            name.startswith("flagquantum/simulation/native_cpu/csrc/")
            and name.endswith(".cpp")
            for name in members
        )
        assert not any(
            name.startswith(("tests/", "examples/", "benchmarks/")) for name in members
        )
        for path in required:
            stream = archive.extractfile(prefix + path)
            assert stream is not None
            assert stream.read() == (ROOT / path).read_bytes()
        stream = archive.extractfile(prefix + "PKG-INFO")
        assert stream is not None
        metadata = Parser().parsestr(stream.read().decode())
    assert metadata["Name"] == "flagquantum"
    assert metadata["Version"] == __version__
    assert "torch<2.15,>=2.10" in metadata.get_all("Requires-Dist", [])
