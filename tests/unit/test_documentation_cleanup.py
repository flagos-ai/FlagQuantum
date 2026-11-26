"""Complete frozen documentation cleanup and owned native publication outputs."""

import csv
import hashlib
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_documentation_commands import original_command
from tools.development import documentation_cleanup as native
from tools.development.notebook_publication import publish_notebooks
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def test_frozen_cleanup_source_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0046")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


@pytest.mark.parametrize("legacy_toc", [False, True])
def test_complete_source_removals_order_and_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_toc: bool
) -> None:
    source = tmp_path / "original"
    target = tmp_path / "native"
    for root in (source, target):
        for name in (
            "first.md",
            "nested/second.md",
            "nested/demo.ipynb",
            "nested/.ipynb_checkpoints/checkpoint.ipynb",
            "nested/.ipynb_checkpoints/resource.png",
            "index.rst",
            "reference/page.html",
            "nested/resource.png",
        ):
            path = root / "docs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Contents\n")
        if legacy_toc:
            (root / "docs/reference/autotoc.rst").write_text("TOC\n")
    with redirect_stdout(io.StringIO()) as expected:
        original_command(
            monkeypatch,
            "clean-stage-2-doc-outputs.py",
            {"git_tools": {"get_git_root": lambda: source}},
            [],
        )
    with redirect_stdout(io.StringIO()) as actual:
        native.clean_generated_outputs(target / "docs")
    assert actual.getvalue().replace(
        str(target), "ROOT"
    ) == expected.getvalue().replace(str(source), "ROOT")
    for root in (source, target):
        assert sorted(
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
        ) == [
            "docs/index.rst",
            "docs/nested/resource.png",
            "docs/reference/page.html",
        ]
    with redirect_stdout(io.StringIO()) as repeated:
        native.clean_generated_outputs(target / "docs")
        native.clean_generated_outputs(target / "absent")
    assert repeated.getvalue() == ""


def test_invalid_checkpoint_preserves_later_files_and_source_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("original", "native"):
        root = tmp_path / name / "docs"
        root.mkdir(parents=True)
        (root / ".ipynb_checkpoints").write_text("Not a directory\n")
        (root / "later.md").write_text("Still present\n")
    with pytest.raises(AssertionError):
        original_command(
            monkeypatch,
            "clean-stage-2-doc-outputs.py",
            {"git_tools": {"get_git_root": lambda: tmp_path / "original"}},
            [],
        )
    with pytest.raises(AssertionError):
        native.clean_generated_outputs(tmp_path / "native/docs")
    assert (tmp_path / "original/docs/later.md").exists()
    assert (tmp_path / "native/docs/later.md").exists()


def test_actual_native_library_publication_cleanup_and_cli_ownership(
    tmp_path: Path,
) -> None:
    output = tmp_path / "docs/components"
    with redirect_stdout(io.StringIO()):
        publish_notebooks(ROOT / "examples/notebooks", output)
    assert len(list(output.glob("*.ipynb"))) == 120
    authored = tmp_path / "docs/development/guide.md"
    authored.parent.mkdir(parents=True)
    authored.write_text("Authored content\n")
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, timeout=15
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.development.documentation_cleanup",
                *arguments,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    for arguments in ((), ("docs",), ("docs/development",), (".",), ("..",)):
        result = run(*arguments)
        assert result.returncode == 2
        assert authored.read_text() == "Authored content\n"
        assert len(list(output.glob("*.ipynb"))) == 120
    result = run("docs/components")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(output.glob("*.ipynb"))
    assert (output / "index.rst").exists()
    assert authored.read_text() == "Authored content\n"
    reference = tmp_path / "docs/reference/flagquantum"
    reference.mkdir(parents=True)
    (reference / "Circuit.md").write_text("Generated reference\n")
    assert run(str(reference)).returncode == 0
    assert not (reference / "Circuit.md").exists()
    staging = tmp_path.parent / (tmp_path.name + "-generated-docs")
    (staging / "reference").mkdir(parents=True)
    toc = staging / "reference/autotoc.rst.inc"
    toc.write_text("Native generated TOC\n")
    assert run(str(staging)).returncode == 0
    assert not toc.exists()


def test_filesystem_failure_preserves_partial_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("original", "native"):
        root = tmp_path / name / "docs"
        (root / ".ipynb_checkpoints").mkdir(parents=True)
        (root / ".ipynb_checkpoints/old.ipynb").write_text("Checkpoint\n")
        (root / "failed.md").write_text("Markdown\n")
        (root / "later.ipynb").write_text("Notebook\n")
    unlink = Path.unlink

    def fail_markdown(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".md":
            raise PermissionError("Injected output removal failure")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_markdown)
    with pytest.raises(PermissionError, match="Injected output removal failure"):
        original_command(
            monkeypatch,
            "clean-stage-2-doc-outputs.py",
            {"git_tools": {"get_git_root": lambda: tmp_path / "original"}},
            [],
        )
    with pytest.raises(PermissionError, match="Injected output removal failure"):
        native.clean_generated_outputs(tmp_path / "native/docs")
    for name in ("original", "native"):
        root = tmp_path / name / "docs"
        assert not (root / ".ipynb_checkpoints").exists()
        assert (root / "failed.md").exists()
        assert (root / "later.ipynb").exists()


def test_cli_rejects_symlink_into_authored_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authored = tmp_path / "docs/development"
    authored.mkdir(parents=True)
    guide = authored / "guide.md"
    guide.write_text("Authored content\n")
    alias = tmp_path / "docs/components"
    alias.symlink_to(authored, target_is_directory=True)
    monkeypatch.setattr(native, "get_git_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as caught:
        native.main([str(alias)])
    assert caught.value.code == 2
    assert guide.exists()
