import json
from pathlib import Path

import pytest

from tools.check_repository_language import violations

pytestmark = pytest.mark.unit


def test_reports_text_location_without_echoing_contents(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    text = chr(0x4E2D)
    path.write_text(f"# Guide\n{text}\n", encoding="utf-8")
    assert violations(path) == [f"{path}:2: Han-script text"]


def test_checks_escaped_notebook_markdown(tmp_path: Path) -> None:
    path = tmp_path / "lesson.ipynb"
    path.write_text(json.dumps({"cells": [{"source": chr(0x4E2D)}]}))
    assert violations(path) == [f"{path}: JSON contains escaped Han-script text"]


def test_accepts_english_and_mathematical_notation(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# State\nThe norm is Σ |α|² = 1.\n", encoding="utf-8")
    assert violations(path) == []


def test_checks_filenames(tmp_path: Path) -> None:
    path = tmp_path / f"{chr(0x4E2D)}.md"
    path.write_text("English content")
    assert violations(path) == [f"{path}: filename contains Han-script characters"]
