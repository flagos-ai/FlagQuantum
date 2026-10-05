from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

import flagquantum
import flagquantum.ecosystem as ecosystem
import flagquantum.remote as remote

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


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


def test_normal_imports_do_not_resolve_vendor_kaiwu_package(tmp_path: Path) -> None:
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
import flagquantum.remote
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
