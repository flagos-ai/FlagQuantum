"""Image landing behavior reuses authenticated personal workshop storage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SCRIPTS = Path(__file__).resolve().parents[2] / "workshops/flagos2026/scripts"


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "workshop_startup", SCRIPTS / "workshop_startup.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_landing_preserves_platform_auth_and_proxy(startup, tmp_path: Path) -> None:
    workshop = tmp_path / "alice/workshop/flagos2026"
    original = [
        "--ServerApp.root_dir",
        "/",
        "--LabApp.default_url=/lab",
        "--ServerApp.token=fixture",
        "--ServerApp.base_url=/cluster/fixture",
        "--port=18888",
        "--allow-root",
    ]
    result = startup.arguments(original, workshop)
    assert original[1] == "/"
    assert result == [
        "--ServerApp.token=fixture",
        "--ServerApp.base_url=/cluster/fixture",
        "--port=18888",
        "--allow-root",
        f"--ServerApp.root_dir={tmp_path / 'alice'}",
        "--ServerApp.default_url=/lab/tree/workshop/flagos2026/notebooks/01_first_circuit.ipynb",
    ]


def test_copies_for_two_users_do_not_overwrite_or_share_outputs(
    startup,
    tmp_path: Path,
) -> None:
    template = tmp_path / "template"
    (template / "notebooks").mkdir(parents=True)
    (template / "notebooks/example.ipynb").write_text("template")
    (template / "outputs").mkdir()
    (template / "outputs/result.json").write_text("private result")
    (template / ".env").write_text("fixture secret")
    workspaces = [tmp_path / name for name in ("alice", "bob")]
    for workspace in workspaces:
        workspace.mkdir()
    alice, bob = [startup.prepare_workshop(template, p) for p in workspaces]
    (alice / "notebooks/example.ipynb").write_text("Alice's edits")
    (alice / "outputs").mkdir()
    (alice / "outputs/result.json").write_text("Alice's result")
    assert startup.prepare_workshop(template, workspaces[0]) == alice
    assert (alice / "notebooks/example.ipynb").read_text() == "Alice's edits"
    assert (bob / "notebooks/example.ipynb").read_text() == "template"
    assert not (bob / "outputs").exists()
    assert not (bob / ".env").exists()


def test_unverified_identity_stops_before_jupyter_launch(
    startup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import ModuleType

    client_module = ModuleType("flagquantum.remote.compute.jiuding")
    client_module.JiudingClient = lambda: object()
    lab_module = ModuleType("jupyterlab.labapp")
    lab_module.main = lambda: pytest.fail("Do not launch in unverified storage")
    monkeypatch.setitem(
        sys.modules, "flagquantum.remote.compute.jiuding", client_module
    )
    monkeypatch.setitem(sys.modules, "jupyterlab.labapp", lab_module)

    def reject_identity(client: object) -> Path:
        raise RuntimeError("No safe platform ID")

    monkeypatch.setattr(startup, "personal_directory", reject_identity)
    with pytest.raises(RuntimeError, match="platform ID"):
        startup.main()
