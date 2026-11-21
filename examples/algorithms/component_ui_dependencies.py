# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependency publication follows Qualtran v0.7.0 dev_tools/ui-export.ipynb.

"""Project existing stored Program dependencies into developer UI node files."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from examples.algorithms.basic_gate_decomposition import decomposition_program
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_expansion import _has_call_decomposition
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tools.development.component_ui_export import (
    component_filename,
    register_interface_score,
)


@dataclass(frozen=True)
class ComponentFunction:
    """Identify a reachable function within its owning catalog Program.

    Owner is the existing root's relative filename. Internal functions retain
    native names and contextual identity. They make no source Bloq metadata or
    cross-root semantic-deduplication claim. The immutable Program remains the
    authoritative body and is excluded from this presentation identity.
    """

    owner: str
    entry: str
    program: ProgramRecord = field(compare=False, hash=False, repr=False)

    @property
    def function(self) -> FunctionRecord:
        return next(f for f in self.program.functions if f.name == self.entry)

    def attributes(self) -> dict[str, object]:
        return {"owner": self.owner, "entry": self.entry}

    def filename(self) -> str:
        return component_filename(self.attributes())

    def score(self) -> object:
        functions = {f.name: f for f in self.program.functions}
        if _has_call_decomposition(self.function, functions):
            return register_score_data(
                component_register_score(self.program, self.entry)
            )
        return register_interface_score(
            Signature(self.function.registers or ()), self.entry
        )


def dependency_graph(
    roots: Iterable[object], *, owner: Callable[[object], str]
) -> dict[object, tuple[object, ...]]:
    """Publish every reachable typed function through the existing graph owner.

    Edges omit multiplicity, matching the UI exporter contract. The Compiler
    retains counts and recognizes mixed/primitive bodies as terminals. Root
    drawings keep their source-specific glyph providers. Descendants use native
    register glyphs and retain native helper boundaries explicitly.
    """
    graph: dict[object, tuple[object, ...]] = {}
    for root in roots:
        program = decomposition_program(root)
        if program is None:
            graph[root] = ()
            continue
        entry = program.functions[0].name
        calls = program_component_call_graph(program, entry)
        names = dict.fromkeys(
            (entry, *(name for edge in calls.direct_calls for name in edge))
        )
        nodes = {
            name: ComponentFunction(owner(root), name, program)
            for name in names
            if name != entry
        }
        for name in names:
            node = root if name == entry else nodes[name]
            graph[node] = tuple(
                nodes[target] for parent, target in calls.direct_calls if parent == name
            )
    return graph
