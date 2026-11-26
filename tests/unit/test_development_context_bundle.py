"""Complete source selection and actual pinned context packing workflows."""

import ast
import csv
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import context_bundle as native
from tools.development import process
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = REFERENCE / "dev_tools/repomix/repomix-driver.py"
PACKER = REFERENCE / "repomix-deps/node_modules/.bin/repomix"


def original_paths() -> list[str]:
    assignment = next(
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "gemini_downselect"
    )
    return list(ast.literal_eval(assignment.value))


def packed(path: Path) -> dict[str, str]:
    return dict(
        re.findall(
            r'<file path="([^"]+)">\n(.*?)\n</file>', path.read_text(), re.DOTALL
        )
    )


def test_full_source_blobs_and_native_selection() -> None:
    count = 0
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in {"Q-0091", "Q-0092"} or row["path"] in original_paths():
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                count += 1
    assert count == 22
    assert len(original_paths()) == len(native.SOURCE_GROUPS) == 20
    assert len({name for name, _ in native.SOURCE_GROUPS}) == 20
    assert all((ROOT / path).is_file() for path in native.source_paths())
    assert len(native.source_paths()) == len(set(native.source_paths()))
    package = json.loads((PACKER.parent.parent / "repomix/package.json").read_text())
    assert package["version"] == "1.18.1"
    assert package["license"] == "MIT"
    lock = json.loads((REFERENCE / "repomix-deps/package-lock.json").read_text())
    assert (
        lock["packages"]["node_modules/repomix"]["integrity"]
        == "sha512-XtV1qD6akJSigQ3fBRex5v+rWcGa4i9cm2FL/M/PEtzO3u1uDiso3CtNbhlZiMIqJstAdy8mj2m1ynOzNnNB3g=="
    )


def test_original_entire_command_and_failure_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, str(ROOT))

    monkeypatch.setattr(subprocess, "run", run)
    loaded = runpy.run_path(str(SOURCE))
    assert calls[0] == (
        ["git", "rev-parse", "--show-toplevel"],
        dict(capture_output=True, universal_newlines=True, check=True),
    )
    assert calls[1] == (
        [
            "npx",
            "repomix",
            "--no-gitignore",
            "--include",
            ",".join([*original_paths(), "docs/**/*.md"]),
            "--output",
            f"dev_tools/repomix/qualtran-repomix-{date.today().isoformat()}.md",
        ],
        dict(check=True, cwd=ROOT),
    )
    assert loaded["includes"] == [*original_paths(), "docs/**/*.md"]
    for failed_index in (0, 1):
        calls.clear()

        def fail(
            command: list[str], *, index: int = failed_index, **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if len(calls) == index:
                raise subprocess.CalledProcessError(7, command)
            return run(command, **kwargs)

        monkeypatch.setattr(subprocess, "run", fail)
        with pytest.raises(subprocess.CalledProcessError):
            runpy.run_path(str(SOURCE))


def test_actual_source_and_native_pack_ignored_docs_and_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=15)
    selected = original_paths()
    groups = tuple(
        (group, (f"native/owner_{i}.py",))
        for i, (group, _) in enumerate(native.SOURCE_GROUPS)
    )
    for i, path in enumerate(selected):
        for relative in (path, groups[i][1][0]):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"VALUE = {i}\n")
    (root / "docs/nested").mkdir(parents=True)
    for name in ("docs/reference.md", "docs/nested/notebook.md", "docs/stale.md"):
        (root / name).write_text(f"# {name}\n")
    (root / "docs/excluded.txt").write_text("excluded")
    (root / ".gitignore").write_text("docs/\n")
    script = root / "dev_tools/repomix/repomix-driver.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(SOURCE.read_bytes())
    binary = tmp_path / "bin"
    binary.mkdir()
    shim = binary / "npx"
    shim.write_text(f'#!/bin/sh\nshift\nexec "{PACKER}" "$@"\n')
    shim.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
    subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    source_output = script.parent / f"qualtran-repomix-{date.today().isoformat()}.md"
    source_files = packed(source_output)
    monkeypatch.setattr(native, "SOURCE_GROUPS", groups)
    output = tmp_path / "native.md"
    assert native.pack_context(root, output=output, executable=str(PACKER)) == 0
    native_files = packed(output)
    mapping = dict(zip(selected, (paths[0] for _, paths in groups), strict=True))
    assert {
        mapping.get(path, path): text for path, text in source_files.items()
    } == native_files
    assert set(native_files) == {
        *mapping.values(),
        "docs/reference.md",
        "docs/nested/notebook.md",
        "docs/stale.md",
    }
    assert native.pack_context(root, output=output, executable=str(PACKER)) == 0
    assert packed(output) == native_files


def test_native_command_status_timeout_missing_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "SOURCE_GROUPS", (("fixture", ("owner.py",)),))
    with pytest.raises(FileNotFoundError, match="owner.py"):
        native.pack_context(tmp_path)
    (tmp_path / "owner.py").write_text("VALUE = 1")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(process, "run", run)
    assert native.pack_context(tmp_path) == 9
    command, kwargs = calls.pop()
    assert command == [
        "repomix",
        "--no-gitignore",
        "--include",
        "owner.py,docs/**/*.md",
        "--output",
        f"tools/development/context/flagquantum-context-{date.today().isoformat()}.md",
    ]
    assert kwargs == dict(
        cwd=tmp_path.resolve(), check=False, timeout=1200, log_run_to_stderr=False
    )
    monkeypatch.setattr(native, "get_git_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as result:
        native.main(["--output", "out.md", "--repomix", "installed-packer"])
    assert result.value.code == 9
    assert calls[-1][0][0] == "installed-packer"

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(process, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        native.pack_context(tmp_path)


def test_native_import_has_no_packing_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Import must not start an external process")

    monkeypatch.setattr(process, "run", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    loaded = runpy.run_path(str(ROOT / "tools/development/context_bundle.py"))
    assert callable(loaded["pack_context"])


def test_real_native_module_cli_and_default_destination(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=15)
    for path in native.source_paths():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.development.context_bundle",
            "--repomix",
            str(PACKER),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    destination = (
        tmp_path
        / "tools/development/context"
        / f"flagquantum-context-{date.today().isoformat()}.md"
    )
    assert set(packed(destination)) == set(native.source_paths())
