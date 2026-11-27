"""Report visible teaching surfaces separately from numerical acceptance.

Run ``python -m tools.synthesis_notebook_audit --write`` after editing a
Notebook. Static observations identify review work; they do not award tutorial
quality, prove execution, or replace the pinned migration inventory.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json"
REPORT = ROOT / "examples/synthesis/NOTEBOOK_UX_AUDIT.md"


@dataclass(frozen=True)
class Observation:
    path: Path
    category: str
    code_cells: int
    qfuncs: int
    synthesis_calls: int
    circuit_calls: int
    helper_modules: tuple[Path, ...]

    @property
    def surface(self) -> str:
        if self.circuit_calls:
            return "Visible Circuit construction"
        if self.qfuncs:
            return "Visible qfunc definition"
        return "Delegated quantum construction"


def dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted_name(node.value)}.{node.attr}"
    return ""


def inspect_notebook(path: Path, category: str) -> Observation:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    trees = [ast.parse(source, filename=str(path)) for source in code]
    qfuncs = synthesis = circuits = 0
    modules: set[Path] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qfuncs += any(
                    dotted_name(decorator).split(".")[-1] == "qfunc"
                    for decorator in node.decorator_list
                )
            elif isinstance(node, ast.Call):
                name = dotted_name(node.func).split(".")[-1]
                synthesis += name == "synthesize"
                circuits += name == "Circuit"
            elif isinstance(node, ast.ImportFrom) and node.module:
                if not node.module.startswith("examples.synthesis."):
                    continue
                for module in (
                    node.module,
                    *(f"{node.module}.{alias.name}" for alias in node.names),
                ):
                    candidate = ROOT / (module.replace(".", "/") + ".py")
                    if candidate.is_file() and "comparisons" not in candidate.parts:
                        modules.add(candidate.relative_to(ROOT))
    return Observation(
        path.relative_to(ROOT),
        category,
        len(code),
        qfuncs,
        synthesis,
        circuits,
        tuple(sorted(modules)),
    )


def observations() -> list[Observation]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows = []
    for group in inventory["groups"]:
        if group["workflow_status"] == "validated":
            notebook = ROOT / group["workflow_validation"]["native_notebook"]
            rows.append(inspect_notebook(notebook, group["category"]))
    return sorted(rows, key=lambda row: str(row.path))


def render(rows: list[Observation]) -> str:
    counts = Counter(row.surface for row in rows)
    lines = [
        "# Notebook teaching-surface audit",
        "",
        "This report covers the accepted migration inventory. It is a static",
        "screen of the editable code cells, not a tutorial-quality certificate.",
        "Visible qfunc definitions may still require better explanations; delegated",
        "construction may be appropriate when teaching an existing algorithm API.",
        "Direct Circuit construction may be a justified comparison. Inspect the",
        "purpose before deciding to rewrite any of these cases.",
        "",
        "Regenerate with `python -m tools.synthesis_notebook_audit --write`.",
        "Acceptance evidence and upstream provenance are not modified.",
        "",
        f"Screened: **{len(rows)}** accepted Notebooks. Editorial completion is tracked",
        "separately in [the improvement plan](NOTEBOOK_UX_PLAN.md).",
        "",
        "| Visible teaching surface | Count |",
        "| --- | ---: |",
    ]
    lines += [f"| {surface} | {count} |" for surface, count in sorted(counts.items())]
    lines += [
        "",
        "## Review priorities",
        "",
        "1. Delegated construction: expose the central quantum function and explain",
        "   what the called public API owns; keep data loading and plotting helpers.",
        "2. Visible Circuit construction: distinguish independent references from",
        "   handwritten application kernels; record actual high-level API gaps.",
        "3. Visible qfunc: explain the quantum data, constraints, returned result",
        "   and limitations. Syntax visibility alone does not satisfy this review.",
        "",
        "The helper column lists direct local imports only. It is a navigation aid,",
        "not a complete dependency graph. Counts recognize literal `qfunc`,",
        "`synthesize` and `Circuit` names; aliased names require manual inspection.",
        "",
        "## Per-Notebook observations",
        "",
        "| Notebook | Surface | Code cells | qfunc definitions | Synthesis calls | Circuit calls | Direct local helpers |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    base = REPORT.parent
    for row in rows:
        link = row.path.relative_to(base.relative_to(ROOT)).as_posix()
        helpers = (
            ", ".join(
                f"[{p.name}]({p.relative_to(base.relative_to(ROOT)).as_posix()})"
                for p in row.helper_modules
            )
            or "—"
        )
        lines.append(
            f"| [{link}]({link}) | {row.surface} | {row.code_cells} | "
            f"{row.qfuncs} | {row.synthesis_calls} | {row.circuit_calls} | {helpers} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    text = render(observations())
    if args.write:
        REPORT.write_text(text, encoding="utf-8")
    elif not REPORT.is_file() or REPORT.read_text(encoding="utf-8") != text:
        raise SystemExit("Teaching-surface report is stale; rerun with --write")
    print("Notebook teaching-surface report verified")


if __name__ == "__main__":
    main()
