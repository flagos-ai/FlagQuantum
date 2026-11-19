"""Every frozen comparison definition has an explicit native disposition."""

import ast
from hashlib import sha1
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_frozen_comparison_definition_disposition() -> None:
    from qualtran.bloqs.arithmetic import comparison

    raw = Path(comparison.__file__).read_bytes()
    audit = (
        Path(__file__).parents[2] / "docs/development/COMPARISON_FILE_AUDIT.md"
    ).read_text()
    digest = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert digest in audit
    assert "bcb17bb25360ed68ce3bc084645db3e929799389" in audit
    count = 0
    for node in ast.parse(raw).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 115
