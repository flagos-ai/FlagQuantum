"""Complete frozen mypy configuration and observable plugin/import behavior."""

import configparser
import csv
import hashlib
import importlib.metadata
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import process, typecheck_command
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = REFERENCE / "mypy-profile-deps"
SOURCE = REFERENCE / "dev_tools/conf/mypy.ini"
PROFILE = Path(typecheck_command.__file__).with_name("retained_source_mypy.ini")
VERSIONS = {
    "mypy": "1.19.1",
    "librt": "0.7.7",
    "mypy-extensions": "1.1.0",
    "numpy": "2.2.6",
    "duet": "0.2.9",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
    "protobuf": "6.33.2",
    "types-protobuf": "6.32.1.20251210",
    "mypy-protobuf": "4.0.0",
    "pathspec": "1.0.1",
    "typing-extensions": "4.15.0",
}
CASES = {
    "duet": (
        "import duet\nasync def impl(x: int) -> str:\n    return str(x)\n"
        'wrapped = duet.sync(impl)\nreveal_type(wrapped)\nwrapped("bad")\n',
        "[arg-type]",
    ),
    "numpy": ("import numpy as np\nreveal_type(np.intp(1))\n", "numpy"),
    "override": (
        "class Parent:\n    def value(self, x: int) -> int:\n        return x\n"
        "class Child(Parent):\n    def value(self, x: str) -> int:\n        return len(x)\n",
        "[override]",
    ),
    "untyped": ('def broken():\n    value: int = "bad"\n', "[assignment]"),
    "redefinition": (
        "items = [1]\nitems = tuple(items)\nreveal_type(items)\n",
        "[assignment]",
    ),
    "sympy": ('import sympy\nreveal_type(sympy.Symbol("x"))\n', "[import-untyped]"),
    "protobuf": (
        "from google.protobuf.message import Message\nreveal_type(Message)\n"
        "Message.missing_native_profile_member\n",
        "[attr-defined]",
    ),
}


def configuration(path: Path) -> configparser.ConfigParser:
    result = configparser.ConfigParser()
    assert result.read(path) == [str(path)]
    return result


def test_complete_source_profiles_and_dependency_blobs() -> None:
    selected = {"Q-0049", "Q-0097", "Q-0106"}
    verified = set()
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in selected:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                verified.add(row["id"])
    assert verified == selected
    source, native = configuration(SOURCE), configuration(PROFILE)
    assert {s: dict(source[s]) for s in source.sections()} == {
        s: dict(native[s]) for s in native.sections()
    }


def test_isolated_dependencies_exactly_match_frozen_lock() -> None:
    distributions = list(importlib.metadata.distributions(path=[str(DEPENDENCIES)]))
    actual = [
        (d.metadata["Name"].lower().replace("_", "-"), d.version) for d in distributions
    ]
    assert len(actual) == len(set(name for name, _ in actual)) == len(VERSIONS)
    assert dict(actual) == VERSIONS
    lock = (REFERENCE / "dev_tools/requirements/envs/mypy.env.txt").read_text()
    for name, version in VERSIONS.items():
        assert f"{name}=={version}\n" in lock


def invoke(
    work: Path, arguments: list[str], native: bool
) -> subprocess.CompletedProcess[str]:
    command = (
        ["-m", "tools.development.typecheck_command", "retained-source"]
        if native
        else ["-m", "mypy"]
    )
    return subprocess.run(
        [sys.executable, *command, *arguments],
        cwd=work,
        env=dict(
            os.environ, PYTHONPATH=os.pathsep.join([str(DEPENDENCIES), str(ROOT)])
        ),
        text=True,
        capture_output=True,
        timeout=180,
    )


@pytest.mark.parametrize("name", CASES)
def test_real_source_native_and_policy_control(tmp_path: Path, name: str) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=15,
    )
    code, diagnostic = CASES[name]
    # Package names here would shadow the actual typing plugins in the subprocess.
    file = tmp_path / f"case_{name}.py"
    file.write_text(code)
    flags = ["--cache-dir=/dev/null", "--no-error-summary", str(file)]
    source = invoke(tmp_path, [f"--config-file={SOURCE}", *flags], False)
    native = invoke(tmp_path, flags, True)
    assert native.returncode == source.returncode, native.stdout + native.stderr
    assert native.stdout == "\033[31m" + source.stdout + "\033[0m"
    assert native.stderr == source.stderr
    control = configuration(SOURCE)
    if name in {"duet", "numpy"}:
        control["mypy"]["plugins"] = ",".join(
            p.strip()
            for p in control["mypy"]["plugins"].split(",")
            if not p.strip().startswith(name)
        )
    elif name == "override":
        control["mypy"]["disable_error_code"] = ""
    elif name == "untyped":
        control["mypy"]["check_untyped_defs"] = "false"
    elif name == "redefinition":
        control["mypy"]["allow_redefinition"] = "false"
    elif name == "sympy":
        control["mypy-sympy.*"]["follow_untyped_imports"] = "false"
    else:
        control["mypy-google.protobuf.*"]["follow_imports"] = "normal"
        control["mypy-google.protobuf.*"]["follow_imports_for_stubs"] = "false"
    path = tmp_path / "control.ini"
    with path.open("w") as stream:
        control.write(stream)
    changed = invoke(tmp_path, [f"--config-file={path}", *flags], False)
    if name == "numpy":
        assert source.returncode == changed.returncode == 0
        assert f"_{struct.calcsize('P') * 8}Bit" in source.stdout
        assert source.stdout != changed.stdout
        assert "_32Bit |" in changed.stdout and "_64Bit" in changed.stdout
    else:
        expected = int(name in {"duet", "untyped"})
        assert source.returncode == expected
        assert changed.returncode == 1 - expected, changed.stdout + changed.stderr
        assert (diagnostic in source.stdout) == bool(expected)
        assert (diagnostic in changed.stdout) == (not expected)
    if name == "duet":
        assert "def (x: builtins.int) -> builtins.str" in source.stdout
        assert "*Any, **Any" in changed.stdout
    elif name == "sympy":
        assert "sympy.core.symbol.Symbol" in source.stdout
        assert 'Revealed type is "Any"' in changed.stdout
    elif name == "protobuf":
        assert 'Revealed type is "Any"' in source.stdout
        assert "google.protobuf.message.Message" in changed.stdout


@pytest.mark.parametrize(
    "module", ["google.fq_missing", "matplotlib.fq_missing", "unlisted_fq_missing"]
)
def test_import_suppression_scope(tmp_path: Path, module: str) -> None:
    file = tmp_path / "case_import.py"
    file.write_text(f"import {module}\n")
    result = invoke(
        tmp_path,
        [f"--config-file={PROFILE}", "--cache-dir=/dev/null", str(file)],
        False,
    )
    assert result.returncode == int(module.startswith("unlisted")), (
        result.stdout + result.stderr
    )
    assert ("[import-not-found]" in result.stdout) == module.startswith("unlisted")


def test_missing_plugin_is_visible(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=15,
    )
    profile = tmp_path / "missing.ini"
    profile.write_text("[mypy]\nplugins = fq_missing_typing_plugin\n")
    file = tmp_path / "case_valid.py"
    file.write_text("value: int = 3\n")
    result = invoke(tmp_path, [f"--config-file={profile}", str(file)], True)
    assert result.returncode == 2
    assert "Error importing plugin" in result.stdout + result.stderr
    assert "fq_missing_typing_plugin" in result.stdout + result.stderr


@pytest.mark.parametrize("status", [0, 1, 2, 17])
def test_profile_mode_process_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    calls = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, status)

    monkeypatch.setattr(process, "run", run)
    arguments = ["--show-error-context", "path with spaces.py"]
    assert (
        typecheck_command.run_typecheck(tmp_path, "retained-source", arguments)
        == status
    )
    assert calls == [
        (
            [sys.executable, "-m", "mypy", f"--config-file={PROFILE}", *arguments],
            dict(cwd=tmp_path, check=False, timeout=1200, log_run_to_stderr=False),
        )
    ]
    assert capsys.readouterr().out == "\033[31m\033[0m"


def test_complete_typing_dependency_declaration_and_closure() -> None:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lines = (
        (REFERENCE / "dev_tools/requirements/deps/mypy.txt").read_text().splitlines()
    )
    requirements = [
        Requirement(line) for line in lines if line and not line.startswith("#")
    ]
    assert [str(requirement) for requirement in requirements] == [
        "mypy~=1.14",
        "mypy-protobuf",
        "sympy",
    ]
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions(path=[str(DEPENDENCIES)])
    }
    for requirement in requirements:
        assert (
            installed[canonicalize_name(requirement.name)].version
            in requirement.specifier
        )
    for distribution in installed.values():
        for line in distribution.requires or []:
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            assert (
                installed[canonicalize_name(requirement.name)].version
                in requirement.specifier
            )
