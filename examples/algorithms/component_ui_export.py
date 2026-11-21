"""Export native component docs, model dependencies and existing Drawer records."""

from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import asdict
from pathlib import Path

from examples.algorithms import register_component_tutorial
from examples.algorithms.register_component_tutorial import WordSwap
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.component_ui_export import (
    component_filename,
    export_component_ui,
)
from tools.development.library_call_graph import library_call_graph
from tools.development.notebook_autogen import NotebookSpec


def make_swap() -> WordSwap:
    """Exchange two two-bit words with six physical CX instructions."""
    return WordSwap(2)


def component_attributes(component: object) -> dict[str, object]:
    """Keep metadata extraction with the example's explicit component domain."""
    if not isinstance(component, (WordSwap, ComponentCost)):
        raise TypeError("UI example requires a WordSwap or its declared CX leaf")
    return asdict(component)


def filename(component: object) -> str:
    return component_filename(component_attributes(component))


def score(component: object) -> object:
    """Give the declared leaf no decomposition diagram, as in the source UI."""
    if isinstance(component, WordSwap):
        program = component.decompose(implemented=True)
        return register_score_data(component_register_score(program, "Swap"))
    if isinstance(component, ComponentCost):
        return None
    raise TypeError("UI example requires a WordSwap or its declared CX leaf")


def declaration(
    component: Hashable, allocator: SymbolAllocator
) -> Mapping[Hashable, int]:
    """Declare logical dependencies independently of physical instruction counts."""
    if isinstance(component, WordSwap):
        return {ComponentCost("CX", 2): 3 * component.width}
    if isinstance(component, ComponentCost):
        return {}
    raise TypeError("UI example requires a WordSwap or its declared CX leaf")


def export_example(destination: Path) -> None:
    """Publish one native component and its complete declared dependency closure."""
    example: ComponentExample[object] = ComponentExample(
        make_swap, "two_bit_word_swap", WordSwap, description=make_swap.__doc__
    )
    documentation = ComponentDocumentation(component_type=WordSwap, examples=(example,))
    spec = NotebookSpec(
        title="Word exchange",
        module=register_component_tutorial,
        component_specs=[documentation],
    )
    graph_examples: list[ComponentExample[Hashable]] = [
        ComponentExample(make_swap, example.name, WordSwap)
    ]
    graph = library_call_graph(graph_examples, declaration)
    export_component_ui(
        destination,
        (("Register components", (spec,)),),
        graph,
        name=lambda component: type(component).__name__,
        attributes=component_attributes,
        filename=filename,
        score=score,
    )
    observed = Counter(
        gate.name for gate in make_swap().make_on((0, 1), (2, 3)).circuit.instructions
    )
    print("Measured primitive counts", dict(observed))
    print("Model primitive estimates", {"cx": sum(graph[make_swap()].values())})


if __name__ == "__main__":
    export_example(Path("component_ui_export"))
