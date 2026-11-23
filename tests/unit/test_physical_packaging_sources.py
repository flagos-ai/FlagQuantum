"""Close original physical build declarations through existing package owners."""

import glob
import runpy
import sys
import types
from collections.abc import Sequence
from hashlib import sha1
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
from setuptools import Extension
from setuptools._distutils.filelist import FileList

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/kickmix"
_BLOBS = {
    "BUILD": "fc704761c0394457990947180e12fceb3b93a250",
    "MANIFEST.in": "3375315df88fc0f97b5e90381f488ed4088e75ce",
    "pyproject.toml": "b756d5fde06082eee77baba5c65836ee8e58b4db",
    "setup.py": "bf3154e46dfeaf7c2d82aeda239cd0b4ce8b80fb",
}


def _source(name: str) -> str:
    raw = (SOURCE / name).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == _BLOBS[name]
    return raw.decode()


def _glob(patterns: list[str], exclude: Sequence[str] = ()) -> list[str]:
    selected = {
        path
        for pattern in patterns
        for path in glob.glob(pattern, root_dir=SOURCE, recursive=True)
        if (SOURCE / path).is_file()
    }
    return sorted(selected.difference(exclude))


@pytest.mark.parametrize("macos", [False, True])
def test_complete_original_build_targets_and_source_partition(macos: bool) -> None:
    targets: dict[str, dict[str, object]] = {}

    def capture(**arguments: object) -> None:
        name = cast(str, arguments["name"])
        assert name not in targets
        targets[name] = arguments

    def select(branches: dict[str, list[str]]) -> list[str]:
        key = "@platforms//os:osx" if macos else "//conditions:default"
        return branches[key]

    load = Mock()
    package = Mock()
    namespace: dict[str, object] = {
        "load": load,
        "package": package,
        "glob": _glob,
        "cc_binary": capture,
        "cc_test": capture,
        "py_wheel": capture,
        "select": select,
    }
    exec(compile(_source("BUILD"), str(SOURCE / "BUILD"), "exec"), namespace)
    load.assert_called_once_with("@rules_python//python:packaging.bzl", "py_wheel")
    package.assert_called_once_with(default_visibility=["//visibility:public"])
    assert set(targets) == {
        "kickmix",
        "kickmix_perf",
        "kickmix_test",
        "kickmix.so",
        "kickmix_dev_wheel",
    }
    common = cast(list[str], namespace["SOURCE_FILES_NO_MAIN"])
    tests = cast(list[str], namespace["TEST_FILES"])
    performance = cast(list[str], namespace["PERF_FILES"])
    bindings = cast(list[str], namespace["PYBIND_FILES"])
    groups = [set(common), set(tests), set(performance), set(bindings)]
    assert all(groups)
    assert all(
        not first & second
        for i, first in enumerate(groups)
        for second in groups[i + 1 :]
    )
    assert set.union(*groups, {"src/kickmix/main.cc"}) == set(
        _glob(["src/**/*.cc", "src/**/*.h", "src/**/*.inl"])
    )
    assert targets["kickmix"]["srcs"] == common + ["src/kickmix/main.cc"]
    assert targets["kickmix_test"]["srcs"] == common + tests
    assert targets["kickmix_perf"]["srcs"] == common + performance
    assert targets["kickmix.so"]["srcs"] == common + bindings
    assert targets["kickmix_test"]["linkopts"] == [
        "-fsanitize=undefined",
        "-fsanitize=address",
    ]
    for name in ("kickmix", "kickmix_test", "kickmix_perf", "kickmix.so"):
        assert targets[name]["includes"] == ["src/"]
        flags = cast(list[str], targets[name]["copts"])
        assert {"-std=c++20", "-fno-strict-aliasing", "-march=native"} <= set(flags)
    assert targets["kickmix.so"]["linkopts"] == (
        ["-Wl,-undefined,dynamic_lookup"] if macos else []
    )
    assert targets["kickmix.so"]["linkshared"] == 1
    assert targets["kickmix.so"]["deps"] == ["@pybind11"]
    assert targets["kickmix_test"]["data"] == _glob(["testdata/**"])
    assert targets["kickmix_perf"]["data"] == _glob(["testdata/**"])
    assert targets["kickmix_dev_wheel"] == {
        "name": "kickmix_dev_wheel",
        "distribution": "kickmix",
        "requires": ["numpy"],
        "strip_path_prefixes": ["kickmix/"],
        "version": "0.0.dev0",
        "deps": [":kickmix.so"],
    }


@pytest.mark.parametrize(
    "system,machine",
    [
        ("Windows", "AMD64"),
        ("Darwin", "arm64"),
        ("Linux", "x86_64"),
        ("Linux", "aarch64"),
    ],
)
def test_complete_original_extension_setup_across_platforms(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source("setup.py")
    pybind = types.ModuleType("pybind11")
    setattr(pybind, "get_include", Mock(return_value="/binding/include"))
    helpers = types.ModuleType("pybind11.setup_helpers")
    compiler = Mock()
    setattr(helpers, "ParallelCompile", compiler)
    monkeypatch.setitem(sys.modules, "pybind11", pybind)
    monkeypatch.setitem(sys.modules, "pybind11.setup_helpers", helpers)
    monkeypatch.chdir(SOURCE)
    with (
        patch("platform.system", return_value=system),
        patch("platform.machine", return_value=machine),
        patch("setuptools.setup") as setup,
    ):
        namespace = runpy.run_path(str(SOURCE / "setup.py"))
    compiler.assert_called_once_with()
    compiler.return_value.install.assert_called_once_with()
    setup.assert_called_once()
    metadata = dict(setup.call_args.kwargs)
    extensions = metadata.pop("ext_modules")
    assert metadata == {
        "name": "kickmix",
        "version": "0.1.dev0",
        "author": "Qualtran Developers",
        "url": "https://github.com/quantumlib/Qualtran",
        "license": "Apache-2.0",
        "description": "Fast construction and simulation of quantum circuits with classical operations, phasing, and X basis demolition measurement.",
        "python_requires": ">=3.10.0",
        "install_requires": ["numpy"],
        "options": {"build": {"build_base": "python_build_kickmix"}},
    }
    assert len(extensions) == 1 and isinstance(extensions[0], Extension)
    extension = extensions[0]
    expected_sources = set(_glob(["src/**/*.cc"])) - set(
        _glob(["src/**/*.test.cc", "src/**/*.perf.cc", "src/**/main.cc"])
    )
    assert (
        extension.sources
        == namespace["RELEVANT_SOURCE_FILES"]
        == sorted(expected_sources)
    )
    assert extension.depends == namespace["HEADER_FILES"]
    assert set(extension.depends) == set(_glob(["src/**/*.h", "src/**/*.inl"]))
    assert extension.include_dirs == ["/binding/include", "src"]
    assert extension.language == "c++" and extension.name == "kickmix"
    flags = (
        ["/std:c++20", "/O2", "/DVERSION_INFO=0.1.dev0"]
        if system == "Windows"
        else [
            "-std=c++20",
            "-fno-strict-aliasing",
            "-fvisibility=hidden",
            "-O3",
            "-g0",
            "-DVERSION_INFO=0.1.dev0",
        ]
    )
    if system != "Windows" and machine == "x86_64":
        flags.append("-mno-avx")
    assert extension.extra_compile_args == flags


def test_complete_original_source_manifest_and_build_environment() -> None:
    paths = _glob(["src/**/*.h", "src/**/*.inl", "src/**/*.cc", "src/**/*_test.py"])
    files = FileList()
    files.set_allfiles(paths)
    for line in _source("MANIFEST.in").splitlines():
        files.process_template_line(line)
    assert set(files.files) == {
        name
        for name in paths
        if not name.endswith((".test.cc", ".test.h", ".perf.cc", ".perf.h", "_test.py"))
    }
    assert tomllib.loads(_source("pyproject.toml")) == {
        "build-system": {
            "requires": ["setuptools", "wheel", "pybind11==2.12.0"],
            "build-backend": "setuptools.build_meta",
        },
        "tool": {
            "cibuildwheel": {
                "test-requires": ["pytest"],
                "test-command": "pytest {package}/src",
            },
            "pytest": {"ini_options": {"pythonpath": ["."]}},
        },
    }
