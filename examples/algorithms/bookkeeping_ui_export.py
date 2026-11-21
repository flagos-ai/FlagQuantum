# Copyright 2023 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Bookkeeping notebook catalog.

"""Export complete bookkeeping interfaces through existing native layout owners.

Run ``python -m examples.algorithms.bookkeeping_ui_export``. This workflow
renders nominal signatures without allocating execution wires.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from sympy import Symbol, symbols

from examples.algorithms.bookkeeping_decomposition import (
    bookkeeping_child as _child_presentation,
)
from examples.algorithms.bookkeeping_decomposition import (
    bookkeeping_program,
    bookkeeping_score,
)
from examples.algorithms.component_ui_dependencies import (
    ComponentFunction,
    dependency_graph,
)
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_partition import (
    partition_signature,
)
from flagquantum.compiler.registers.layout.partition import partition_width
from flagquantum.compiler.registers.layout.register_cast import cast_signature
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
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


_AUTO_PARTITION_LAYOUTS = {
    "auto_partition": AutoPartition(
        (("x", (("ctrl", 1), ("x", 1))), ("y", (("y", 1),)))
    ),
    "auto_partition_unused": AutoPartition(
        (
            ("x", (("ctrl", 1), ("x", 1), (None, 1))),
            ("y", (("y", 1),)),
            ("z", ((None, 2),)),
        )
    ),
}


def _instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    dtype = unsigned_register_type(4)
    for identifier, factory in (
        ("split", split_word_function),
        ("join", join_word_function),
    ):
        function = factory(identifier, dtype)
        assert function.registers is not None
        result[identifier] = (
            identifier.title(),
            Signature(function.registers),
            {"dtype": {"bitsize": 4}},
            factory.__module__,
        )
    n1, n2 = symbols("n1 n2")
    parts = (
        Register("y1", opaque_register_type(n1)),
        Register("y2", opaque_register_type(n2)),
    )
    for identifier, reverse in (("split2", False), ("join2", True)):
        result[identifier] = (
            identifier.title(),
            partition_signature(parts, reverse=reverse),
            {"n1": n1, "n2": n2},
            partition_signature.__module__,
        )
    n = Symbol("n")
    for identifier, operation, side in (
        ("alloc", "Allocate", RegisterSide.RIGHT),
        ("free", "Free", RegisterSide.LEFT),
    ):
        result[identifier] = (
            operation,
            Signature((Register("reg", unsigned_register_type(n), side=side),)),
            {"dtype": {"bitsize": n}, "dirty": False},
            "flagquantum.compiler.registers.component_builder.ComponentProgramBuilder",
        )
    shaped = (
        Register("xx", opaque_register_type(2), (2, 3)),
        Register("yy", opaque_register_type(37)),
    )
    result["partition"] = (
        "Partition",
        partition_signature(shaped),
        {
            "n": 49,
            "regs": tuple(
                {
                    "name": port.name,
                    "dtype": {"bitsize": port.bitsize},
                    "_shape": port.shape,
                    "side": port.side,
                }
                for port in shaped
            ),
            "partition": True,
        },
        partition_signature.__module__,
    )
    for identifier, layout in _AUTO_PARTITION_LAYOUTS.items():
        ports = Signature(
            Register(
                name,
                opaque_register_type(
                    partition_width(
                        tuple(
                            (str(index), width)
                            for index, (_, width) in enumerate(group)
                        )
                    )
                ),
            )
            for name, group in layout.groups
        )
        # Retain native control values and unused segments instead of serializing
        # a foreign controlled-child object and losing its NumPy control scalar.
        result[identifier] = (
            "AutoPartition",
            ports,
            {
                "child": {"operation": "Swap", "bitsize": 1, "control_values": (1,)},
                "groups": layout.groups,
                "left_only": layout.left_only,
            },
            AutoPartition.__module__,
        )
    result["cast"] = (
        "Cast",
        cast_signature(signed_register_type(32), fixed_point_register_type(32, 32)),
        {
            "inp_dtype": {"bitsize": 32},
            "out_dtype": {"bitsize": 32, "num_frac": 32, "signed": False},
            "shape": (),
            "allow_quantum_to_classical": False,
        },
        cast_signature.__module__,
    )
    result["always_and"] = (
        "Always",
        Signature(
            (
                Register("ctrl", quantum_bit_type(), (2,)),
                Register("target", quantum_bit_type(), side=RegisterSide.RIGHT),
            )
        ),
        {"subbloq": {"cv1": 1, "cv2": 1, "uncompute": False}},
        "flagquantum.compiler.registers.primitives.always.Always",
    )
    return result


INSTANCES = _instances()
DOCUMENTS = (
    ("Bookkeeping", "Split / Join", "register_split_join", "Split", ("split",)),
    ("Bookkeeping", "Split / Join", "register_split_join", "Join", ("join",)),
    ("Bookkeeping", "Split2 / Join2", "register_split2_join2", "Split2", ("split2",)),
    ("Bookkeeping", "Split2 / Join2", "register_split2_join2", "Join2", ("join2",)),
    ("Bookkeeping", "Alloc / Free", "register_allocate_free", "Allocate", ("alloc",)),
    ("Bookkeeping", "Alloc / Free", "register_allocate_free", "Free", ("free",)),
    ("Bookkeeping", "Partition", "register_partition", "Partition", ("partition",)),
    (
        "Bookkeeping",
        "Partition",
        "register_partition",
        "AutoPartition",
        ("auto_partition", "auto_partition_unused"),
    ),
    ("Bookkeeping", "Cast", "register_cast", "Cast", ("cast",)),
    ("Bookkeeping", "Always", "register_always", "Always", ("always_and",)),
)


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Bookkeeping UI requires a registered native example identifier"
        )
    return INSTANCES[component]


def name(component: object) -> str:
    return (
        "ComponentFunction"
        if isinstance(component, ComponentFunction)
        else _instance(component)[0]
    )


def signature(component: object) -> Signature:
    return _instance(component)[1]


def attributes(component: object) -> Mapping[str, object]:
    return (
        component.attributes()
        if isinstance(component, ComponentFunction)
        else _instance(component)[2]
    )


def filename(component: object) -> str:
    return (
        component.filename()
        if isinstance(component, ComponentFunction)
        else component_filename(attributes(component))
    )


def score(component: object) -> object:
    if isinstance(component, ComponentFunction):
        return component.score()
    record_score = bookkeeping_score(
        name(component),
        signature(component),
        _AUTO_PARTITION_LAYOUTS.get(str(component)),
    )
    return (
        record_score
        if record_score is not None
        else register_interface_score(signature(component), name(component))
    )


def decomposition(component: object) -> ProgramRecord | None:
    """Reuse complete native wrapper graphs and retain source atomic refusals."""
    return bookkeeping_program(
        name(component),
        signature(component),
        _AUTO_PARTITION_LAYOUTS.get(str(component)),
    )


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = _instance(documentation.examples[0].make())[3]
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "AutoPartition and Always retain complete stored wrapper graphs. "
        "Original atomic operations retain their typed interfaces. "
        "Execution primitives and resource models retain separate evidence. "
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
                import_line="from examples.algorithms.bookkeeping_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete bookkeeping metadata and fresh interfaces."""
    export_component_ui(
        destination,
        sections(),
        dependency_graph(
            INSTANCES,
            owner=lambda component: name(component) + "/" + filename(component),
            program=decomposition,
            child_presentation=_child_presentation,
        ),
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )


if __name__ == "__main__":
    export_example(Path("bookkeeping_ui_export"))
