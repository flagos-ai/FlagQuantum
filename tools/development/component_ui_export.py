# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/ui-export.ipynb.

"""Export supplied native component documentation and declared graph artifacts."""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from sympy import Symbol

from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.testing.component_examples import ComponentDocumentation
from tools.development.docstrings import get_markdown_docstring
from tools.development.notebook_autogen import NotebookSpec


def _documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return documentation.component_type.__name__


def _documentation_text(documentation: ComponentDocumentation[object]) -> str:
    return "\n".join(get_markdown_docstring(documentation.component_type))


class ComponentUIEncoder(json.JSONEncoder):
    """Retain source scalar spellings for native metadata and Drawer dictionaries."""

    def default(self, value: object) -> object:
        if isinstance(value, Symbol):
            return f"Symbol({value})"
        if isinstance(value, complex):
            return f"{value.real}+{value.imag}i"
        if isinstance(value, RegisterSide):
            return {
                RegisterSide.LEFT: "Side.LEFT",
                RegisterSide.RIGHT: "Side.RIGHT",
                RegisterSide.THRU: "Side.THRU",
            }.get(value, "Side.THRU")
        return "NOT_SERIALIZABLE"


def component_filename(
    attributes: Mapping[str, object],
    prefix: str = "",
    *,
    encoder: type[json.JSONEncoder] = ComponentUIEncoder,
) -> str:
    """Hash sorted parameter pairs with the source's non-security MD5 convention.

    The domain supplies parameter metadata and any wrapper prefix. No graph,
    callable or runtime object is inspected or implicitly serialized here.
    """
    pairs = [[key, attributes[key]] for key in sorted(attributes)]
    payload = json.dumps(pairs, cls=encoder)
    return (
        prefix
        + hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()
        + ".json"
    )


def register_interface_score(ports: Signature, name: str) -> object:
    """Draw nominal directional ports without admitting executable storage."""
    # Preserve nominal register metadata and different LEFT/RIGHT lifetimes.
    widths: dict[str, SymbolicInt] = {}
    inputs, outputs = [], []
    for port in ports:
        for index in port.all_idxs():
            label = port.name + ("[" + ",".join(map(str, index)) + "]" if index else "")
            if port.side == RegisterSide.RIGHT:
                label = "out:" + label
            widths[label] = port.bitsize
            if port.side != RegisterSide.RIGHT:
                inputs.append(label)
            if port.side != RegisterSide.LEFT:
                outputs.append(label)
    step = RegisterStep(
        name,
        tuple(inputs),
        tuple(outputs),
        input_registers=tuple(ports.lefts()),
        output_registers=tuple(ports.rights()),
    )
    return register_score_data(
        layout_register_score(
            widths,
            inputs,
            (step,),
            input_registers=tuple(ports.lefts()),
            output_registers=tuple(ports.rights()),
            outputs=outputs,
        )
    )


def export_component_ui(
    destination: Path,
    sections: Sequence[tuple[str, Sequence[NotebookSpec]]],
    graph: Mapping[object, Iterable[object]],
    *,
    name: Callable[[object], str],
    attributes: Callable[[object], Mapping[str, object]],
    filename: Callable[[object], str],
    score: Callable[[object], object],
    encoder: type[json.JSONEncoder] = ComponentUIEncoder,
    documentation_name: Callable[
        [ComponentDocumentation[object]], str
    ] = _documentation_name,
    documentation_text: Callable[
        [ComponentDocumentation[object]], str
    ] = _documentation_text,
) -> None:
    """Publish navigation, class documentation and all supplied graph nodes.

    Domains supply identities, metadata, declared graph and Drawer data through
    existing owners. Graph edges describe dependencies and omit multiplicity,
    matching the source UI export, not a physical resource report. Navigation
    is refreshed each run. Existing class docs and example files are retained.
    Score providers return None when their domain cannot provide a diagram.
    Ordinary score-provider failures yield a null diagram. Process cancellation,
    other provider failures and filesystem failures propagate, retaining earlier files.
    Explicit documentation providers support domains whose operations share a
    native IR or resource-call class. Defaults retain class names and docstrings.
    """
    navigation = {
        section: {
            spec.title: [
                {
                    "name": documentation_name(documentation),
                    "examples": [
                        {
                            "name": example.name,
                            "example_docstring": example.description,
                            "filename": filename(example.make()),
                        }
                        for example in documentation.examples
                    ],
                }
                for documentation in spec.component_specs
            ]
            for spec in specifications
        }
        for section, specifications in sections
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "navigation.json").write_text(json.dumps(navigation, indent=2))
    for _, specifications in sections:
        for spec in specifications:
            for documentation in spec.component_specs:
                folder = destination / documentation_name(documentation)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "docs.txt"
                if not path.is_file():
                    path.write_text(documentation_text(documentation))
    for component, children in graph.items():
        folder = destination / name(component)
        path = folder / filename(component)
        if path.is_file():
            continue
        label = str(component)
        metadata = attributes(component)
        try:
            diagram = score(component)
        except Exception:
            # Diagram availability is an explicit UI boundary, as in the source.
            diagram = None
        record = {
            "name": label,
            "attrs": metadata,
            "msd": diagram,
            "callees": [
                {"name": name(child), "filename": filename(child)} for child in children
            ],
        }
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, cls=encoder))
