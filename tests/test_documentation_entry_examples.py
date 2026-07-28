from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(
    r"^```python\n(.*?)^```",
    flags=re.MULTILINE | re.DOTALL,
)


def _execute_first_python_block(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    match = PYTHON_FENCE.search(path.read_text(encoding="utf-8"))

    assert match is not None
    namespace: dict[str, object] = {"__name__": "__documentation_example__"}
    exec(compile(match.group(1), f"{relative_path}:first-python-block", "exec"), namespace)
    return namespace


@pytest.mark.integration
def test_runtime_result_contract_entry_example_executes() -> None:
    namespace = _execute_first_python_block("docs/reference/RUNTIME_RESULT_CONTRACT.md")

    assert namespace["result"].plan is not None


@pytest.mark.integration
def test_hybrid_runtime_entry_example_executes() -> None:
    namespace = _execute_first_python_block("docs/architecture/HYBRID_RUNTIME_ARCHITECTURE.md")

    assert namespace["loss"].grad_fn is not None
