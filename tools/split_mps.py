#!/usr/bin/env python
"""One-shot decomposition of the native MPS simulation module."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/simulation/mps.py"
paths = {
    "models": SOURCE.with_name("mps_models.py"),
    "planning": SOURCE.with_name("mps_planning_mixin.py"),
    "state": SOURCE.with_name("mps_state.py"),
    "factorization": SOURCE.with_name("mps_factorization.py"),
    "execution": SOURCE.with_name("mps_execution.py"),
}
if any(path.exists() for path in paths.values()):
    raise SystemExit("MPS split targets already exist")
text = SOURCE.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
defs = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
model_nodes = [node for node in defs if node.end_lineno <= 233]
state_node = next(node for node in defs if node.name == "MPSState")
factor_nodes = [node for node in defs if 1454 <= node.lineno <= 1759]
execution_nodes = [node for node in defs if node.lineno >= 1762]
planning_methods = [
    node
    for node in state_node.body
    if isinstance(node, ast.FunctionDef) and 289 <= node.lineno <= 444
]
header_end = min(node.lineno for node in defs) - 1
header = "".join(lines[:header_end])


def start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(item.lineno for item in decorators)]) - 1


def render(nodes: list[ast.AST]) -> str:
    return "".join(lines[start(nodes[0]) : nodes[-1].end_lineno])


def imports(module: str, nodes: list[ast.AST]) -> str:
    return (
        f"from .{module} import (\n"
        + "".join(f"    {node.name},\n" for node in nodes)
        + ")\n\n"
    )


models_import = imports("mps_models", model_nodes)
factor_import = imports("mps_factorization", factor_nodes)
planning_body = render(planning_methods)
paths["models"].write_text(header + render(model_nodes), encoding="utf-8")
paths["planning"].write_text(
    header + models_import + "class MPSPlanningMixin:\n" + planning_body,
    encoding="utf-8",
)

state_text = "".join(lines[start(state_node) : state_node.end_lineno])
method_start = start(planning_methods[0]) - start(state_node)
method_end = planning_methods[-1].end_lineno - start(state_node)
state_lines = state_text.splitlines(keepends=True)
state_text = "".join(state_lines[:method_start] + state_lines[method_end:])
state_text = state_text.replace(
    "class MPSState:", "class MPSState(MPSPlanningMixin):", 1
)
paths["state"].write_text(
    header
    + models_import
    + "from .mps_planning_mixin import MPSPlanningMixin\n"
    + factor_import
    + state_text,
    encoding="utf-8",
)
paths["factorization"].write_text(
    header + models_import + render(factor_nodes), encoding="utf-8"
)
paths["execution"].write_text(
    header
    + models_import
    + factor_import
    + "from .mps_state import MPSState\n\n"
    + render(execution_nodes),
    encoding="utf-8",
)
SOURCE.write_text(
    '''"""Compatibility facade for native MPS simulation."""

from importlib import import_module
from typing import Any

_MODULES = (
    "flagquantum.simulation.mps_models",
    "flagquantum.simulation.mps_factorization",
    "flagquantum.simulation.mps_state",
    "flagquantum.simulation.mps_execution",
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


__all__ = (
    "CompiledMPSOperation", "CompiledMPSProgram", "MPSAdaptiveBondPlan",
    "MPSAdaptiveRunResult", "MPSBondProfile", "MPSConfig",
    "MPSLocalRefinementPlan", "MPSMonteCarloResult", "MPSState",
    "MPSTruncationRecord", "run_mps", "run_mps_adaptive",
    "run_noisy_mps", "run_noisy_mps_trajectory",
)
''',
    encoding="utf-8",
)
