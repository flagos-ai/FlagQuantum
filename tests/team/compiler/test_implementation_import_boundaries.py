"""Compiler implementations depend directly on their owning stage modules."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPILER = Path(__file__).resolve().parents[3] / "flagquantum" / "compiler"


def test_implementations_do_not_depend_on_expert_forwarding_modules() -> None:
    forwarding_modules = {
        f"flagquantum.compiler.{path.stem}"
        for path in COMPILER.glob("*.py")
        if path.stem not in {"__init__", "pipeline"}
    }
    assert "flagquantum.compiler.routing" in forwarding_modules
    violations: list[str] = []
    implementations = sorted(
        path for path in COMPILER.rglob("*.py") if path.parent != COMPILER
    )
    assert implementations
    for path in implementations:
        relative = path.relative_to(COMPILER).with_suffix("")
        parts = relative.parts[:-1]
        package = ".".join(("flagquantum", "compiler", *parts))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = resolve_name("." * node.level + (node.module or ""), package)
                imported = [module]
                if module == "flagquantum.compiler":
                    imported.extend(f"{module}.{alias.name}" for alias in node.names)
            else:
                continue
            for module in imported:
                if module in forwarding_modules:
                    violations.append(
                        f"{path.relative_to(COMPILER)}:{node.lineno} {module}"
                    )
    assert not violations, "Import the owning stage module directly:\n" + "\n".join(
        violations
    )
