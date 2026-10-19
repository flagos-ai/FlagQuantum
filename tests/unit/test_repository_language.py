import json
from pathlib import Path

import pytest

from tools import check_repository_language
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


@pytest.fixture
def localized_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(check_repository_language, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "workshops/flagos2026/notebooks_cn/01_first_circuit.ipynb"
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize("ensure_ascii", (True, False))
def test_localized_markdown_prose_is_allowed(
    localized_path: Path, ensure_ascii: bool
) -> None:
    document = {
        "cells": [
            {"cell_type": "markdown", "source": [chr(0x4E2D)], "metadata": {}},
            {"cell_type": "code", "source": ["print(1)"], "outputs": []},
        ],
        "metadata": {},
    }
    localized_path.write_text(json.dumps(document, ensure_ascii=ensure_ascii))
    before = localized_path.read_bytes()
    assert violations(localized_path) == []
    assert localized_path.read_bytes() == before


@pytest.mark.parametrize("ensure_ascii", (True, False))
@pytest.mark.parametrize("location", ("code", "output", "cell_metadata", "metadata"))
def test_localization_does_not_exempt_code_outputs_or_metadata(
    localized_path: Path, ensure_ascii: bool, location: str
) -> None:
    han = chr(0x4E2D)
    markdown = {"cell_type": "markdown", "source": han, "metadata": {}}
    code = {"cell_type": "code", "source": "print(1)", "outputs": []}
    document = {"cells": [markdown, code], "metadata": {}}
    if location == "code":
        code["source"] = han
    elif location == "output":
        code["outputs"] = [{"output_type": "stream", "text": han}]
    elif location == "cell_metadata":
        markdown["metadata"] = {"title": han}
    else:
        document["metadata"] = {"title": han}
    localized_path.write_text(json.dumps(document, ensure_ascii=ensure_ascii))
    assert violations(localized_path) == [
        f"{localized_path}: Han-script text outside localized Markdown source"
    ]


@pytest.mark.parametrize(
    "relative",
    (
        "workshops/flagos2026/notebooks/01_first_circuit.ipynb",
        "workshops/flagos2026/notebooks_cn/nested/01_first_circuit.ipynb",
        "workshops/flagos2026/notebooks_cn/custom.ipynb",
        "workshops/flagos2026/notebooks_cn/README.md",
        "another/notebooks_cn/01_first_circuit.ipynb",
    ),
)
def test_localization_scope_is_exact(localized_path: Path, relative: str) -> None:
    path = check_repository_language.REPOSITORY_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"cells": [{"cell_type": "markdown", "source": chr(0x4E2D)}]},
            ensure_ascii=False,
        )
    )
    assert violations(path)


@pytest.mark.parametrize(
    "text",
    (
        "{",
        "[]",
        '{"cells": {}}',
        '{"cells": [1]}',
        '{"cells": [{"cell_type": "markdown", "source": 1}]}',
    ),
)
def test_localized_notebook_structure_fails_closed(
    localized_path: Path, text: str
) -> None:
    localized_path.write_text(text)
    assert violations(localized_path)


def test_localized_path_cannot_exempt_an_external_symlink(
    localized_path: Path, tmp_path: Path
) -> None:
    external = tmp_path / "external.ipynb"
    external.write_text(
        json.dumps({"cells": [{"cell_type": "markdown", "source": chr(0x4E2D)}]})
    )
    localized_path.symlink_to(external)
    assert violations(localized_path)


@pytest.mark.parametrize("fence", ("```", "~~~~"))
@pytest.mark.parametrize("closed", (True, False))
def test_localized_fenced_code_remains_english(
    localized_path: Path, fence: str, closed: bool
) -> None:
    source = f"{fence}python\nprint('{chr(0x4E2D)}')\n"
    if closed:
        source += fence
    localized_path.write_text(
        json.dumps({"cells": [{"cell_type": "markdown", "source": source}]})
    )
    assert violations(localized_path) == [
        f"{localized_path}: Han-script text in localized code example"
    ]
