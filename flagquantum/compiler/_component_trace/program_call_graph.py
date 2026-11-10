"""Count stored typed component calls through the existing resource graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ...core.ir.program import ProgramRecord
from ..registers.component_expansion import _has_call_decomposition
from ..registers.component_graph import _call_entry
from .call_graph import SymbolicCallGraph, symbolic_call_graph


def program_component_call_graph(
    record: ProgramRecord,
    entry: str = "main",
    *,
    keep: Callable[[str], bool] | None = None,
    max_depth: int | None = None,
) -> SymbolicCallGraph[str]:
    """Count actual typed-call multiplicities with source resource depth semantics.

    Primitive/mixed bodies are terminal at this component boundary. This counts
    stored calls, independently of formula estimates and emitted gate counts.
    All definitions are verified before counting. Unsupported entries raise
    ValueError. Keep/depth policies delegate to the existing resource graph.
    """
    _call_entry(record, entry)
    functions = {f.name: f for f in record.functions}
    declared = {
        function.name: Counter(op.callee or "" for op in function.blocks[0].operations)
        for function in record.functions
        if function.registers is not None
        and _has_call_decomposition(function, functions)
    }
    return symbolic_call_graph(entry, declared, keep=keep, max_depth=max_depth)
