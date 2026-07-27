#!/usr/bin/env python
"""One-shot decomposition of distributed audit policy."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/audit/engine.py"
MPS = SOURCE.with_name("mps_readiness.py")
RELEASE = SOURCE.with_name("release_policy.py")
if MPS.exists() or RELEASE.exists():
    raise SystemExit("audit split targets already exist")
text = SOURCE.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
defs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
mps_nodes = [node for node in defs if 824 <= node.lineno <= 1552]
release_nodes = [node for node in defs if 1555 <= node.lineno <= 1980]


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


header_end = min(node.lineno for node in defs) - 1
header = "".join(lines[:header_end])
base_nodes = [node for node in defs if node.end_lineno < 824]
schema_imports = """from .errors import DistributedScalabilityError
from .schema import (
    DistributedEvidenceContract,
    DistributedScalabilityAudit,
    DistributedTransportEvidence,
    MPSBackwardReadinessGate,
    StatevectorTrainingClaimabilityGate,
)

"""
MPS.write_text(
    header + schema_imports + imports("engine", base_nodes) + render(mps_nodes),
    encoding="utf-8",
)
RELEASE.write_text(
    header
    + schema_imports
    + imports("engine", base_nodes)
    + imports("mps_readiness", mps_nodes)
    + render(release_nodes),
    encoding="utf-8",
)
mps_start, mps_end = start(mps_nodes[0]), mps_nodes[-1].end_lineno
release_start, release_end = start(release_nodes[0]), release_nodes[-1].end_lineno
remaining = (
    "".join(lines[:mps_start])
    + "".join(lines[mps_end:release_start])
    + "".join(lines[release_end:])
)
SOURCE.write_text(remaining, encoding="utf-8")
