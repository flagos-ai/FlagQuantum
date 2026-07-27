#!/usr/bin/env python
"""One-shot extraction of the statevector forward executor."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/backends/statevector/forward.py"
TARGET = SOURCE.with_name("forward_executor.py")
text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
node = next(
    item
    for item in tree.body
    if isinstance(item, ast.FunctionDef)
    and item.name == "execute_torch_distributed_statevector"
)
if TARGET.exists():
    raise SystemExit("forward executor target already exists")
lines = text.splitlines(keepends=True)
implementation = "".join(lines[node.lineno - 1 : node.end_lineno])
prefix = text[
    : lines[: node.lineno - 1].__len__() and sum(map(len, lines[: node.lineno - 1]))
]
prior = [
    item.name
    for item in tree.body
    if isinstance(item, (ast.FunctionDef, ast.ClassDef))
    and item.end_lineno < node.lineno
]
imports = (
    "from .forward import (\n" + "".join(f"    {name},\n" for name in prior) + ")\n\n"
)
header = '''"""Torch-distributed statevector forward execution loop."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import CircuitIR, Instruction
from .state import DistributedStatevectorPlan
''' + imports
wrapper = '''def execute_torch_distributed_statevector(
    *args: Any, **kwargs: Any
) -> TorchDistributedStatevectorResult:
    """Compatibility entrypoint delegated to the bounded executor module."""
    from .forward_executor import execute_torch_distributed_statevector as execute

    return execute(*args, **kwargs)
'''
SOURCE.write_text(prefix + wrapper + "\n", encoding="utf-8")
TARGET.write_text(header + implementation + "\n", encoding="utf-8")
