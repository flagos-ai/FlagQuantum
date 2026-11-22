"""Complete frozen documentation Makefile and actual Sphinx consumers."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sphinx_environment(tmp_path: Path) -> dict[str, str]:
    """Expose the accepted optional Python tools through their original commands."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    script = f"#!{sys.executable}\n" + """
import json
import os
import runpy
import sys
from pathlib import Path
name = Path(sys.argv[0]).name
if "FQ_DOC_TRACE" in os.environ:
    with Path(os.environ["FQ_DOC_TRACE"]).open("a") as stream:
        stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
    raise SystemExit(int(os.environ.get("FQ_DOC_EXIT", "0")))
sys.path[:0] = json.loads(os.environ["FQ_DOC_TOOL_PATHS"])
module = {
    "sphinx-build": "sphinx.cmd.build",
    "sphinx-autobuild": "sphinx_autobuild",
}[name]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""
    for name in ("sphinx-build", "sphinx-autobuild", "alternate-build"):
        path = binaries / name
        path.write_text(script)
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": os.pathsep.join((str(binaries), "/usr/bin", "/bin")),
        "FQ_DOC_TOOL_PATHS": json.dumps(sys.path),
    }


def test_complete_makefile_and_build_guide_source_blobs() -> None:
    expected = {
        "docs/Makefile": "4e92d9f146ba0ce21dd9f0b7302212d423a454d7",
        "dev_tools/how-to-build-the-docs.md": "921749792909a29d8c8ba4a6594970db882d9fff",
    }
    for name, blob in expected.items():
        raw = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    assert (ROOT / "docs/Makefile").read_bytes() == (
        REFERENCE / "docs/Makefile"
    ).read_bytes()


@pytest.mark.parametrize(
    "target",
    [
        None,
        "help",
        "html",
        "clean",
        "auto",
        "autobuild",
        "live",
        "livehtml",
        "auto-dev",
        "unrecognized",
    ],
)
def test_complete_source_native_target_commands(
    target: str | None, tmp_path: Path, sphinx_environment: dict[str, str]
) -> None:
    trace = tmp_path / "commands.jsonl"
    environment = {
        **sphinx_environment,
        "FQ_DOC_TRACE": str(trace),
        "SPHINXOPTS": "-W --keep-going",
    }
    results = []
    for root in (REFERENCE, ROOT):
        trace.unlink(missing_ok=True)
        shutil.copyfile(root / "docs/Makefile", tmp_path / "Makefile")
        command = ["make", "O=-j 2"]
        if target is not None:
            command.append(target)
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        assert len(calls) == 1
        results.append(calls[0])
    assert results[0] == results[1]
    selected = "help" if target is None else target
    options = ["-W", "--keep-going"]
    if selected in ("auto", "autobuild", "live", "livehtml", "auto-dev"):
        forced = ["-a"] if selected == "auto-dev" else []
        assert results[0] == [
            "sphinx-autobuild",
            *forced,
            ".",
            "_build/html",
            *options,
            "--open-browser",
            "-j",
            "auto",
            "-j",
            "2",
        ]
    else:
        assert results[0] == [
            "sphinx-build",
            "-M",
            selected,
            ".",
            "_build",
            *options,
            "-j",
            "2",
        ]


def test_original_build_override_and_child_failure_propagation(
    tmp_path: Path, sphinx_environment: dict[str, str]
) -> None:
    trace = tmp_path / "commands.jsonl"
    for root in (REFERENCE, ROOT):
        shutil.copyfile(root / "docs/Makefile", tmp_path / "Makefile")
        result = subprocess.run(
            ["make", "html"],
            cwd=tmp_path,
            env={
                **sphinx_environment,
                "FQ_DOC_TRACE": str(trace),
                "FQ_DOC_EXIT": "7",
                "SPHINXBUILD": "alternate-build",
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 2
        assert "Error 7" in result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    assert calls == [["alternate-build", "-M", "html", ".", "_build"]] * 2


@pytest.mark.parametrize("missing", ["sphinx-build", "sphinx-autobuild"])
def test_original_preflight_requires_both_default_executables(
    missing: str, tmp_path: Path, sphinx_environment: dict[str, str]
) -> None:
    (tmp_path / "bin" / missing).unlink()
    trace = tmp_path / "commands.jsonl"
    for root in (REFERENCE, ROOT):
        shutil.copyfile(root / "docs/Makefile", tmp_path / "Makefile")
        result = subprocess.run(
            [
                "make",
                "help",
                "SPHINXBUILD=alternate-build",
            ],
            cwd=tmp_path,
            env={**sphinx_environment, "FQ_DOC_TRACE": str(trace)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 2
        assert f'Cannot find program "{missing}"' in result.stderr
        assert not trace.exists()


@pytest.mark.parametrize("profile", ["source", "native"])
def test_actual_make_help_html_incremental_and_clean(
    profile: str, tmp_path: Path, sphinx_environment: dict[str, str]
) -> None:
    project = tmp_path / "documentation source"
    project.mkdir()
    origin = REFERENCE if profile == "source" else ROOT
    shutil.copyfile(origin / "docs/Makefile", project / "Makefile")
    (project / "conf.py").write_text("project = 'Build acceptance'\n")
    page = project / "index.rst"
    page.write_text("First document\n==============\n\nInitial content.\n")
    output = tmp_path / "rendered docs"
    for target in ("help", "html", "html", "clean", "unrecognized"):
        result = subprocess.run(
            ["make", target, f"BUILDDIR={output}", "SPHINXOPTS=-W --keep-going"],
            cwd=project,
            env=sphinx_environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if target == "unrecognized":
            assert result.returncode != 0
            assert "not registered" in result.stdout + result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
        if target == "help":
            assert "html" in result.stdout
        elif target == "html":
            html = (output / "html/index.html").read_text()
            assert (
                "Initial content."
                if "First document" in page.read_text()
                else "Changed content."
            ) in html
            page.write_text("Changed document\n================\n\nChanged content.\n")
        elif target == "clean":
            assert not (output / "html/index.html").exists()
    assert "Changed content." in page.read_text()
