#!/usr/bin/env python
"""One-shot split of statevector contracts, planning, and local execution."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/backends/statevector/state.py"
MODELS = SOURCE.with_name("models.py")
PLANNING = SOURCE.with_name("planning.py")
EXECUTION = SOURCE.with_name("local_execution.py")

text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
definitions = [
    node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
]
model_defs = [node for node in definitions if node.lineno <= 594]
planning_defs = [node for node in definitions if 597 <= node.lineno <= 1227]
execution_defs = [node for node in definitions if node.lineno >= 1230]
if any(path.exists() for path in (MODELS, PLANNING, EXECUTION)):
    raise SystemExit("statevector split targets already exist")

header_end = text.index("@dataclass(frozen=True)")
header = text[:header_end]


def block(nodes: list[ast.AST]) -> str:
    start = min(node.lineno for node in nodes) - 1
    end = max(node.end_lineno for node in nodes)
    return "".join(text.splitlines(keepends=True)[start:end])


def imports(module: str, nodes: list[ast.AST]) -> str:
    return (
        f"from .{module} import (\n"
        + "".join(f"    {node.name},\n" for node in nodes)
        + ")\n\n"
    )


models_text = header.replace(
    "Distributed statevector planning and diagnostics.",
    "Distributed statevector contracts and immutable records.",
    1,
) + block(model_defs)
planning_text = (
    header.replace(
        "Distributed statevector planning and diagnostics.",
        "Distributed statevector topology and memory planning.",
        1,
    )
    + imports("models", model_defs)
    + block(planning_defs)
)
execution_text = (
    header.replace(
        "Distributed statevector planning and diagnostics.",
        "Local and torch-distributed statevector execution helpers.",
        1,
    )
    + imports("models", model_defs)
    + imports("planning", planning_defs)
    + block(execution_defs)
)
facade = (
    '"""Compatibility facade for distributed statevector APIs."""\n\n'
    + imports("models", model_defs)
    + imports("planning", planning_defs)
    + imports("local_execution", execution_defs)
)

MODELS.write_text(models_text, encoding="utf-8")
PLANNING.write_text(planning_text, encoding="utf-8")
EXECUTION.write_text(execution_text, encoding="utf-8")
SOURCE.write_text(facade, encoding="utf-8")
