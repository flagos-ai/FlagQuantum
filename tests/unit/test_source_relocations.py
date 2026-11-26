"""Source moves preserve frozen identities without hiding new declarations."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tools.architecture.source_relocations import (
    relocation_errors,
    source_identity,
    source_location,
)

pytestmark = pytest.mark.unit


def _moved_package(tmp_path: Path) -> Path:
    package = tmp_path / "flagquantum"
    package.mkdir()
    (package / "original.py").write_text(
        "from flagquantum.moved import Model as Model\n"
    )
    (package / "moved.py").write_text(
        "class Model:\n    def run(self, wire):\n        return wire\n"
    )
    manifest = tmp_path / "docs/development/migration/independent-structure-moves.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "moves": [
                    {
                        "old": "flagquantum/original.py",
                        "new": "flagquantum/moved.py",
                        "symbols": ["Model"],
                    }
                ]
            }
        )
    )
    return package


def test_known_source_keeps_its_ledger_identity_and_unknown_names_stay_visible(
    tmp_path: Path,
) -> None:
    package = _moved_package(tmp_path)
    assert relocation_errors(tmp_path) == ()
    census = importlib.import_module("tools.census_wire_vocabulary")
    before = census.census(package)
    assert [site.identifier for site in before.canonical] == [
        "flagquantum/original.py::Model.run::wire"
    ]
    with (package / "moved.py").open("a") as stream:
        stream.write("\ndef unregistered(wires):\n    return wires\n")
    after = census.census(package)
    assert "flagquantum/moved.py::unregistered::wires" in {
        site.identifier for site in after.canonical
    }
    assert (
        source_location(tmp_path, "flagquantum/original.py::Model.run::wire")
        == package / "moved.py"
    )
    assert source_identity(package, "flagquantum/moved.py", "unregistered") == (
        "flagquantum/moved.py"
    )


def test_retaining_the_original_implementation_is_not_a_valid_facade(
    tmp_path: Path,
) -> None:
    _moved_package(tmp_path)
    (tmp_path / "flagquantum/original.py").write_text("class Model:\n    pass\n")
    assert any(
        "still defines implementations" in error
        for error in relocation_errors(tmp_path)
    )


def test_missing_new_implementation_fails_closed(tmp_path: Path) -> None:
    package = _moved_package(tmp_path)
    (package / "moved.py").unlink()
    assert any(
        "missing relocated implementation" in error
        for error in relocation_errors(tmp_path)
    )


def test_keyword_guard_resolves_absolute_relative_and_module_imports() -> None:
    import ast

    census = importlib.import_module("tools.census_wire_vocabulary")
    aliases = census._import_aliases(
        ast.parse(
            "from flagquantum.core.measurement import samples as sample_request\n"
            "from ..measurement import counts\n"
            "from . import schema\n"
            "import flagquantum.core.measurement\n"
        ),
        "flagquantum.core.operators",
    )
    assert aliases == {
        "sample_request": "flagquantum.core.measurement.samples",
        "counts": "flagquantum.core.measurement.counts",
        "schema": "flagquantum.core.operators.schema",
        "flagquantum": "flagquantum",
    }
