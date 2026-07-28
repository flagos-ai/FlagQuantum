from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(
    r"^```python\n(.*?)^```",
    flags=re.MULTILINE | re.DOTALL,
)


@pytest.mark.integration
def test_readme_golden_path_executes() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = PYTHON_FENCE.search(text)

    assert match is not None
    namespace = {"__name__": "__readme_golden_path__"}
    exec(compile(match.group(1), "README.md:first-python-block", "exec"), namespace)

    assert namespace["training"].completed_steps == 10
    assert namespace["result"].plan is not None
    assert namespace["package"].backend.provider == "local"
