# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependency publication follows Qualtran v0.7.0 dev_tools/ui-export.ipynb.

"""Project existing stored Program dependencies into developer UI node files."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property

from examples.algorithms.basic_gate_child_decomposition import child_decomposition
from examples.algorithms.basic_gate_decomposition import decomposition_program
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_expansion import _has_call_decomposition
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import (
    RegisterWireSymbolSupplier,
    register_score_data,
)
from tools.development.component_ui_export import (
    component_filename,
    register_interface_score,
)

ChildPresentation = tuple[str, ProgramRecord, RegisterWireSymbolSupplier]
ChildPresentationProvider = Callable[
    [ProgramRecord, str, object], ChildPresentation | None
]


@dataclass(frozen=True)
class ComponentFunction:
    """Identify a reachable function within its owning catalog Program.

    Owner is the existing root's relative filename. Internal functions retain
    native names and contextual identity. Recognized complete definitions reuse
    their native constructor labels and source-specific drawing symbols. The
    immutable Program remains the authoritative body and is excluded from this
    presentation identity. Cross-root semantic deduplication is unqualified.
    """

    owner: str
    entry: str
    program: ProgramRecord = field(compare=False, hash=False, repr=False)
    component: object = field(default=None, compare=False, hash=False, repr=False)

    child_presentation: ChildPresentationProvider = field(
        default=child_decomposition, compare=False, hash=False, repr=False
    )

    @cached_property
    def presentation(
        self,
    ) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
        return self.child_presentation(self.program, self.entry, self.component)

    @property
    def function(self) -> FunctionRecord:
        return next(f for f in self.program.functions if f.name == self.entry)

    def attributes(self) -> dict[str, object]:
        result: dict[str, object] = {"owner": self.owner, "entry": self.entry}
        if self.presentation is not None:
            result["operation"] = self.presentation[0]
        return result

    def filename(self) -> str:
        return component_filename({"owner": self.owner, "entry": self.entry})

    def score(self) -> object:
        if self.presentation is not None:
            _, record, symbols = self.presentation
            return register_score_data(
                component_register_score(record, record.functions[0].name),
                register_wire_symbol=symbols,
            )
        functions = {f.name: f for f in self.program.functions}
        if _has_call_decomposition(self.function, functions):
            return register_score_data(
                component_register_score(self.program, self.entry)
            )
        return register_interface_score(
            Signature(self.function.registers or ()), self.entry
        )


def dependency_graph(
    roots: Iterable[object],
    *,
    owner: Callable[[object], str],
    program: Callable[[object], ProgramRecord | None] = decomposition_program,
    child_presentation: ChildPresentationProvider = child_decomposition,
) -> dict[object, tuple[object, ...]]:
    """Publish every reachable typed function through the existing graph owner.

    Edges omit multiplicity, matching the UI exporter contract. The Compiler
    retains counts and recognizes mixed/primitive bodies as terminals. Root
    drawings keep their source-specific glyph providers. Qualified descendants
    reuse matching constructor symbols. Other descendants use native register
    glyphs and retain native helper boundaries explicitly.
    """
    graph: dict[object, tuple[object, ...]] = {}
    for root in roots:
        record = program(root)
        if record is None:
            graph[root] = ()
            continue
        entry = record.functions[0].name
        calls = program_component_call_graph(record, entry)
        names = dict.fromkeys(
            (entry, *(name for edge in calls.direct_calls for name in edge))
        )
        nodes = {
            name: ComponentFunction(owner(root), name, record, root, child_presentation)
            for name in names
            if name != entry
        }
        for name in names:
            node = root if name == entry else nodes[name]
            graph[node] = tuple(
                nodes[target] for parent, target in calls.direct_calls if parent == name
            )
    return graph
