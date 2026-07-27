#!/usr/bin/env python
"""One-shot extraction of JAX MPS kernels and gate primitives."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/backends/jax/kernel.py"
MPS = SOURCE.with_name("mps_kernel.py")
GATES = SOURCE.with_name("gate_primitives.py")
text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
lines = text.splitlines(keepends=True)
if MPS.exists() or GATES.exists():
    raise SystemExit("JAX split targets already exist")


def nodes_between(first: int, last: int) -> list[ast.AST]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and first <= node.lineno <= last
    ]


def bounds(nodes: list[ast.AST]) -> tuple[int, int]:
    return min(node.lineno for node in nodes) - 1, max(
        node.end_lineno for node in nodes
    )


mps_nodes = nodes_between(432, 1231)
gate_nodes = nodes_between(1496, 1834)
mps_start, mps_end = bounds(mps_nodes)
gate_start, gate_end = bounds(gate_nodes)
header_end = next(
    node.lineno - 1
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
)
header = "".join(lines[:header_end])
prior = [
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.end_lineno <= 429
]
prior_import = (
    "from .kernel import (\n" + "".join(f"    {name},\n" for name in prior) + ")\n\n"
)
gate_import = (
    "from .gate_primitives import (\n"
    + "".join(f"    {node.name},\n" for node in gate_nodes)
    + ")\n\n"
)
mps_import = (
    "from .mps_kernel import (\n"
    + "".join(f"    {node.name},\n" for node in mps_nodes)
    + ")\n\n"
)
GATES.write_text(
    header.replace("JAX quantum kernels", "JAX gate primitives", 1)
    + prior_import
    + "".join(lines[gate_start:gate_end]),
    encoding="utf-8",
)
MPS.write_text(
    header.replace("JAX quantum kernels", "JAX MPS quantum kernels", 1)
    + prior_import
    + gate_import
    + "".join(lines[mps_start:mps_end]),
    encoding="utf-8",
)
remaining = (
    "".join(lines[:mps_start])
    + mps_import
    + "".join(lines[mps_end:gate_start])
    + gate_import
    + "".join(lines[gate_end:])
)
SOURCE.write_text(remaining, encoding="utf-8")
