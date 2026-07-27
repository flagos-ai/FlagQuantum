#!/usr/bin/env python
"""One-shot decomposition of the tensor-network simulation module."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/simulation/tensor.py"
TARGETS = {
    "models": SOURCE.with_name("tensor_models.py"),
    "state": SOURCE.with_name("tensor_state.py"),
    "planning": SOURCE.with_name("tensor_contraction.py"),
    "execution": SOURCE.with_name("tensor_execution.py"),
}
text = SOURCE.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
defs = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
groups = {
    "models": [node for node in defs if node.lineno <= 611],
    "state": [node for node in defs if 614 <= node.lineno <= 820],
    "planning": [node for node in defs if 823 <= node.lineno <= 1797],
    "execution": [node for node in defs if node.lineno >= 1800],
}
if any(path.exists() for path in TARGETS.values()):
    raise SystemExit("tensor-network split targets already exist")
header_end = min(node.lineno for node in defs) - 1
header = "".join(lines[:header_end])


def node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(item.lineno for item in decorators)]) - 1


def render(nodes: list[ast.AST]) -> str:
    return "".join(lines[node_start(nodes[0]) : nodes[-1].end_lineno])


def imports(module: str, nodes: list[ast.AST]) -> str:
    return (
        f"from .{module} import (\n"
        + "".join(f"    {node.name},\n" for node in nodes)
        + ")\n\n"
    )


TARGETS["models"].write_text(header + render(groups["models"]), encoding="utf-8")
TARGETS["state"].write_text(
    header + imports("tensor_models", groups["models"]) + render(groups["state"]),
    encoding="utf-8",
)
TARGETS["planning"].write_text(
    header
    + imports("tensor_models", groups["models"])
    + imports("tensor_state", groups["state"])
    + render(groups["planning"]),
    encoding="utf-8",
)
TARGETS["execution"].write_text(
    header
    + imports("tensor_models", groups["models"])
    + imports("tensor_state", groups["state"])
    + imports("tensor_contraction", groups["planning"])
    + render(groups["execution"]),
    encoding="utf-8",
)
SOURCE.write_text(
    '''"""Compatibility facade for tensor-network simulation."""

from importlib import import_module
from typing import Any

_MODULES = (
    "flagquantum.simulation.tensor_models",
    "flagquantum.simulation.tensor_state",
    "flagquantum.simulation.tensor_contraction",
    "flagquantum.simulation.tensor_execution",
)


def __getattr__(name: str) -> Any:
    for module_name in _MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _MODULES:
        names.update(dir(import_module(module_name)))
    return sorted(names)
''',
    encoding="utf-8",
)
