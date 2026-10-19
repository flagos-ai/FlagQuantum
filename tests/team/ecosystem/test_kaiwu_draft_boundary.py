from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import flagquantum
import flagquantum.ecosystem as ecosystem
import flagquantum.remote as remote

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ECOSYSTEM_KAIWU = ROOT / "flagquantum/ecosystem/kaiwu"
REMOTE_KAIWU = ROOT / "flagquantum/remote/kaiwu"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            imported.append(node.args[0].value)
    return tuple(imported)


def test_kaiwu_draft_is_not_exported_from_stable_or_parent_facades() -> None:
    assert "kaiwu" not in flagquantum.__all__
    assert not any("kaiwu" in name.lower() for name in ecosystem.__all__)
    assert not any("kaiwu" in name.lower() for name in remote.__all__)


def test_kaiwu_draft_has_no_capability_maturity_entry() -> None:
    with (ROOT / "capability-maturity.toml").open("rb") as stream:
        capabilities = tomllib.load(stream)["capabilities"]

    assert not any(
        token in capability.lower()
        for capability in capabilities
        for token in ("kaiwu", "qboson", "qdiffusion")
    )


def test_draft_imports_do_not_resolve_vendor_kaiwu_package(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys

class RejectKaiwu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kaiwu" or fullname.startswith("kaiwu."):
            raise RuntimeError("vendor Kaiwu import attempted")
        return None

sys.meta_path.insert(0, RejectKaiwu())
import flagquantum
import flagquantum.ecosystem
import flagquantum.ecosystem.kaiwu
import flagquantum.remote
import flagquantum.remote.kaiwu
assert not any(name == "kaiwu" or name.startswith("kaiwu.") for name in sys.modules)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-B", "-s", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_kaiwu_source_dependencies_preserve_ecosystem_remote_split() -> None:
    ecosystem_vendor_imports: list[str] = []
    remote_layer_leaks: list[str] = []
    vendor_import_sites: list[str] = []

    for path in sorted(ECOSYSTEM_KAIWU.glob("*.py")):
        for module in _imports(path):
            if module == "kaiwu" or module.startswith("kaiwu."):
                ecosystem_vendor_imports.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {module}"
                )

    for path in sorted(REMOTE_KAIWU.glob("*.py")):
        for module in _imports(path):
            root = module.split(".", 1)[0]
            if (
                module.startswith("flagquantum.ecosystem")
                or module.startswith("ecosystem")
                or root == "torch"
                or module.startswith("kaiwu.torch_plugin")
            ):
                remote_layer_leaks.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {module}"
                )
            if module == "kaiwu" or module.startswith("kaiwu."):
                vendor_import_sites.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {module}"
                )

    assert ecosystem_vendor_imports == []
    assert remote_layer_leaks == []
    assert vendor_import_sites == ["flagquantum/remote/kaiwu/sdk.py -> kaiwu"]
