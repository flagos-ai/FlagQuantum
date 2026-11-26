"""Complete optional documentation inputs and actual local rebuild consumer."""

import hashlib
import importlib
import importlib.metadata
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.unit.notebook_data import installed_jupyter_data
from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def test_complete_documentation_input_and_installed_direct_requirements() -> None:
    from packaging.requirements import Requirement

    path = REFERENCE / "dev_tools/requirements/deps/docs.txt"
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "fa1c3a966986532f9d1e81715e45468b53960c55"
    )
    requirements = [
        Requirement(line)
        for line in raw.decode().splitlines()
        if line and not line.startswith("#")
    ]
    assert [str(requirement) for requirement in requirements] == [
        "ipywidgets",
        "nbconvert",
        "nbformat",
        "sphinx",
        "sphinx-autobuild",
        "pydata-sphinx-theme",
        "myst-nb",
        "tensorflow-docs~=2023.5.24",
        "griffe",
    ]
    for requirement in requirements:
        distribution = importlib.metadata.distribution(requirement.name)
        assert distribution.version in requirement.specifier
        assert any("license" in str(path).lower() for path in distribution.files or [])
    for distribution in importlib.metadata.distributions(
        path=[str(REFERENCE / "site-autobuild-deps")]
    ):
        for line in distribution.requires or []:
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            assert importlib.metadata.version(requirement.name) in requirement.specifier


def test_real_widget_state_and_notebook_export(monkeypatch: pytest.MonkeyPatch) -> None:
    import ipywidgets

    nbformat = importlib.import_module("nbformat")
    nbconvert = importlib.import_module("nbconvert")

    monkeypatch.setenv("JUPYTER_PATH", str(installed_jupyter_data()))
    widget = ipywidgets.IntSlider(value=3, min=0, max=10)
    try:
        assert widget.get_state()["value"] == 3
        widget.value = 7
        assert widget.get_state()["value"] == 7
    finally:
        widget.close()
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Native component documentation"),
            nbformat.v4.new_code_cell(
                "print(7)",
                execution_count=1,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="7\n")],
            ),
        ]
    )
    markdown, _ = nbconvert.MarkdownExporter().from_notebook_node(notebook)
    assert isinstance(markdown, str)
    assert "Native component documentation" in markdown
    assert "print(7)" in markdown
    assert "7" in markdown


def test_actual_autobuild_serves_rebuilt_native_documentation(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "conf.py").write_bytes((ROOT / "docs/conf.py").read_bytes())
    for name in ("_static", "_templates"):
        shutil.copytree(ROOT / "docs" / name, source / name)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/logo_flagquantum.png").write_bytes(
        (ROOT / "assets/logo_flagquantum.png").read_bytes()
    )
    page = source / "index.md"
    page.write_text("# Initial native document\n\nFirst content.\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log_path = tmp_path / "autobuild.log"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sphinx_autobuild",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                str(source),
                str(tmp_path / "html"),
            ],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            start_new_session=os.name == "posix",
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for expected in ("Initial native document", "Rebuilt native document"):
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        with opener.open(
                            f"http://127.0.0.1:{port}/index.html", timeout=1
                        ) as response:
                            html = response.read().decode()
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                        continue
                    if expected in html:
                        break
                    time.sleep(0.1)
                else:
                    pytest.fail(log_path.read_text())
                if expected == "Initial native document":
                    page.write_text("# Rebuilt native document\n\nChanged content.\n")
            assert "Changed content." in html
        finally:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
    assert "build succeeded" in log_path.read_text()
