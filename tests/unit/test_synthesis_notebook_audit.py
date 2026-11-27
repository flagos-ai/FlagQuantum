"""Static screening distinguishes code from prose without claiming acceptance."""

import json
from pathlib import Path

import pytest

from tools import synthesis_notebook_audit as audit

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("source", "surface", "count"),
    [
        (
            "# @qfunc and fq.Circuit() are only a comment",
            "Delegated quantum construction",
            0,
        ),
        ("@synth.qfunc\ndef main():\n    pass", "Visible qfunc definition", 1),
        ("reference = fq.Circuit(2)", "Visible Circuit construction", 0),
    ],
)
def test_screen_inspects_executable_syntax_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    surface: str,
    count: int,
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "lesson.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["@qfunc def invented()"]},
                    {"cell_type": "code", "source": [source]},
                ]
            }
        )
    )
    row = audit.inspect_notebook(path, "tutorials")
    assert row.surface == surface
    assert row.qfuncs == count
    assert row.code_cells == 1


def test_screen_covers_current_inventory_without_claiming_quality() -> None:
    rows = audit.observations()
    assert len(rows) == 212
    assert len({row.path for row in rows}) == 212
    assert {row.category for row in rows} == {
        "algorithms",
        "applications",
        "community",
        "functions",
        "tutorials",
    }
    assert audit.REPORT.read_text() == audit.render(rows)
