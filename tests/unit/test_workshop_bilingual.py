"""Keep localized teaching prose aligned with the English executable examples."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.check_repository_language import CHINESE_NOTEBOOK_NAMES, HAN, violations

pytestmark = pytest.mark.unit
WORKSHOP = Path(__file__).resolve().parents[2] / "workshops/flagos2026"


def test_chinese_edition_has_exactly_the_six_approved_examples() -> None:
    assert {
        path.name for path in (WORKSHOP / "notebooks_cn").iterdir()
    } == CHINESE_NOTEBOOK_NAMES


@pytest.mark.parametrize("name", sorted(CHINESE_NOTEBOOK_NAMES))
def test_translation_preserves_executable_cells_and_examples(name: str) -> None:
    english = json.loads((WORKSHOP / "notebooks" / name).read_text(encoding="utf-8"))
    path = WORKSHOP / "notebooks_cn" / name
    chinese = json.loads(path.read_text(encoding="utf-8"))
    assert violations(path) == []
    assert chinese["metadata"] == english["metadata"]
    assert len(chinese["cells"]) == len(english["cells"])
    for original, translated in zip(english["cells"], chinese["cells"], strict=True):
        assert original["cell_type"] == translated["cell_type"]
        if original["cell_type"] == "code":
            assert original == translated
            assert translated["execution_count"] is None
            assert translated["outputs"] == []
        else:
            source = "".join(translated["source"])
            assert HAN.search(source)
            assert translated["metadata"] == original["metadata"]
            assert re.findall(r"```python\n(.*?)```", source, re.DOTALL) == re.findall(
                r"```python\n(.*?)```", "".join(original["source"]), re.DOTALL
            )
    assert f"../notebooks/{name}" in "".join(chinese["cells"][0]["source"])
    assert f"notebooks_cn/{name}" in (WORKSHOP / "README.md").read_text()
