# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Optimization notebook catalog.

"""Export complete k-XOR and Kikuchi example interfaces from native declarations.

Run ``python -m examples.algorithms.optimization_ui_export``. Existing algorithm
owners retain their physical and resource-only boundaries.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from flagquantum.algorithms.kikuchi_adjacency import (
    KikuchiNonZeroIndex,
    example_kikuchi_nonzero_index,
)
from flagquantum.algorithms.kikuchi_block_encoding import (
    KikuchiHamiltonian,
    example_kikuchi_hamiltonian,
    symbolic_kikuchi_hamiltonian,
)
from flagquantum.algorithms.kikuchi_guiding_state import (
    GuidingState,
    ScopeGuidingState,
    example_guiding_state,
    example_scope_guiding_state,
    symbolic_count_guiding_state,
    symbolic_guiding_state,
    symbolic_scope_guiding_state,
)
from flagquantum.algorithms.kikuchi_matrix import (
    KikuchiMatrixEntry,
    example_kikuchi_matrix_entry,
    symbolic_kikuchi_matrix_entry,
)
from flagquantum.algorithms.kxor_instance import KXorInstance
from flagquantum.algorithms.kxor_loading import (
    LoadConstraintScopes,
    example_scope_loader,
    symbolic_scope_loader,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.value_dimensions import SymbolicSequenceLength
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.component_ui_export import (
    component_filename,
    export_component_ui,
    register_interface_score,
)
from tools.development.notebook_autogen import NotebookSpec

_Interface: TypeAlias = tuple[str, Signature, Mapping[str, object], str]
_Operation: TypeAlias = (
    LoadConstraintScopes
    | ScopeGuidingState
    | GuidingState
    | KikuchiNonZeroIndex
    | KikuchiMatrixEntry
    | KikuchiHamiltonian
)


def _instance_parameters(instance: KXorInstance) -> Mapping[str, object]:
    constraints = instance.constraints
    return {
        "n": instance.n,
        "k": instance.k,
        "constraints": (
            {"n": constraints.length}
            if isinstance(constraints, SymbolicSequenceLength)
            else tuple({"S": clause.S, "b": clause.b} for clause in constraints)
        ),
        "max_rhs": instance.max_rhs,
    }


def _instances() -> dict[str, _Interface]:
    operations: tuple[tuple[str, str, _Operation], ...] = (
        ("load_scopes_symb", "LoadConstraintScopes", symbolic_scope_loader()),
        ("load_scopes", "LoadConstraintScopes", example_scope_loader()),
        (
            "simple_guiding_state_symb",
            "SimpleGuidingState",
            symbolic_scope_guiding_state(),
        ),
        ("simple_guiding_state", "SimpleGuidingState", example_scope_guiding_state()),
        ("guiding_state_symb_c", "GuidingState", symbolic_count_guiding_state()),
        ("guiding_state_symb", "GuidingState", symbolic_guiding_state()),
        ("guiding_state", "GuidingState", example_guiding_state()),
        (
            "kikuchi_nonzero_index",
            "KikuchiNonZeroIndex",
            example_kikuchi_nonzero_index(),
        ),
        (
            "kikuchi_matrix_entry_symb",
            "KikuchiMatrixEntry",
            symbolic_kikuchi_matrix_entry(),
        ),
        ("kikuchi_matrix_entry", "KikuchiMatrixEntry", example_kikuchi_matrix_entry()),
        ("kikuchi_matrix", "KikuchiHamiltonian", example_kikuchi_hamiltonian()),
        ("kikuchi_matrix_symb", "KikuchiHamiltonian", symbolic_kikuchi_hamiltonian()),
    )
    result: dict[str, _Interface] = {}
    for identifier, label, operation in operations:
        params: dict[str, object] = {"inst": _instance_parameters(operation.inst)}
        if isinstance(operation, (ScopeGuidingState, GuidingState)):
            params["eps"] = operation.eps
        if isinstance(
            operation,
            (GuidingState, KikuchiNonZeroIndex, KikuchiMatrixEntry, KikuchiHamiltonian),
        ):
            params["ell"] = operation.ell
        if isinstance(operation, GuidingState):
            params["amplitude_good_part"] = operation.amplitude_good_part
        if isinstance(operation, (KikuchiMatrixEntry, KikuchiHamiltonian)):
            params["entry_bitsize"] = operation.entry_bitsize
        if isinstance(operation, (KikuchiNonZeroIndex, KikuchiHamiltonian)):
            params["s"] = operation.s
        result[identifier] = (
            label,
            operation.signature,
            params,
            type(operation).__module__,
        )
    return result


INSTANCES = _instances()
DOCUMENTS = (
    (
        "Optimization",
        "kXOR: Instance load Oracles",
        "kxor_instance_loading",
        "LoadConstraintScopes",
        ("load_scopes_symb", "load_scopes"),
    ),
    (
        "Optimization",
        "Planted Noisy kXOR - Kikuchi Guiding State",
        "kikuchi_guiding_state",
        "SimpleGuidingState",
        ("simple_guiding_state_symb", "simple_guiding_state"),
    ),
    (
        "Optimization",
        "Planted Noisy kXOR - Kikuchi Guiding State",
        "kikuchi_guiding_state",
        "GuidingState",
        ("guiding_state_symb_c", "guiding_state_symb", "guiding_state"),
    ),
    (
        "Optimization",
        "Planted Noisy kXOR: Kikuchi Adjacency List",
        "kikuchi_adjacency_list",
        "KikuchiNonZeroIndex",
        ("kikuchi_nonzero_index",),
    ),
    (
        "Optimization",
        "Planted Noisy kXOR: Kikuchi Adjacency Matrix",
        "kikuchi_adjacency_matrix",
        "KikuchiMatrixEntry",
        ("kikuchi_matrix_entry_symb", "kikuchi_matrix_entry"),
    ),
    (
        "Optimization",
        "Planted Noisy kXOR: Block-encoding the Kikuchi Matrix",
        "kikuchi_block_encoding",
        "KikuchiHamiltonian",
        ("kikuchi_matrix", "kikuchi_matrix_symb"),
    ),
)


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Optimization UI requires a registered native example identifier"
        )
    return INSTANCES[component]


def name(component: object) -> str:
    return _instance(component)[0]


def signature(component: object) -> Signature:
    return _instance(component)[1]


def attributes(component: object) -> Mapping[str, object]:
    return _instance(component)[2]


def filename(component: object) -> str:
    return component_filename(attributes(component))


def score(component: object) -> object:
    return register_interface_score(signature(component), name(component))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = _instance(documentation.examples[0].make())[3]
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "This diagram shows the atomic typed interface. Complete decomposition "
        "graphs, glyphs, execution and resource estimates retain separate evidence. "
        "Large and symbolic instances allocate no execution wires.\n"
    )


def _identity(identifier: str) -> object:
    return identifier


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, dict[str, NotebookSpec]] = {}
    for section, title, stem, _, identifiers in DOCUMENTS:
        notebooks = grouped.setdefault(section, {})
        if title not in notebooks:
            notebooks[title] = NotebookSpec(
                title=title,
                module=sys.modules[__name__],
                component_specs=[],
                directory=str(Path(__file__).resolve().parents[1] / "notebooks"),
                path_stem=stem,
            )
        notebooks[title].component_specs.append(
            ComponentDocumentation(
                component_type=object,
                examples=tuple(
                    ComponentExample(partial(_identity, identifier), identifier, object)
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.optimization_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete optimization metadata and fresh interfaces."""
    export_component_ui(
        destination,
        sections(),
        dict.fromkeys(INSTANCES, ()),
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )


if __name__ == "__main__":
    export_example(Path("optimization_ui_export"))
