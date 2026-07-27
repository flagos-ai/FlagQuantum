#!/usr/bin/env python
"""One-shot extraction of statevector reverse-mode adjoint mechanics."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/backends/statevector/reverse.py"
TARGET = SOURCE.with_name("reverse_adjoint.py")
text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
moved = [
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and 365 <= node.lineno <= 1477
]
if TARGET.exists():
    raise SystemExit("reverse adjoint target already exists")
lines = text.splitlines(keepends=True)
start = min(node.lineno for node in moved) - 1
end = max(node.end_lineno for node in moved)
block = "".join(lines[start:end])
prior = [
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.end_lineno < start + 1
]
header_end = next(
    node.lineno - 1
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
)
header = "".join(lines[:header_end]).replace(
    "PyTorch-native sharded statevector reverse mode.",
    "Explicit sharded statevector adjoint mechanics.",
    1,
)
prior_import = (
    "from .reverse import (\n" + "".join(f"    {name},\n" for name in prior) + ")\n\n"
)
moved_import = (
    "from .reverse_adjoint import (\n"
    + "".join(f"    {node.name},\n" for node in moved)
    + ")\n\n"
)
TARGET.write_text(header + prior_import + block + "\n", encoding="utf-8")
SOURCE.write_text(
    "".join(lines[:start]) + moved_import + "".join(lines[end:]), encoding="utf-8"
)
