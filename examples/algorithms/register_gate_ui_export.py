# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Basic Gates notebook catalog.

"""Export register-gate interfaces using the existing domain-owned instances.

Run ``python -m examples.algorithms.register_gate_ui_export``. This metadata
workflow covers the remaining six Basic Gates notebook entries. Execution,
workspace, effects and model queries remain with their existing domain owners.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path

from sympy import Symbol

from examples.algorithms import basic_gate_ui_export as basic
from examples.algorithms.and_notebook import and_signature
from examples.algorithms.basic_gate_decomposition import decomposition_score
from examples.algorithms.component_ui_dependencies import (
    ComponentFunction,
    dependency_graph,
)
from examples.algorithms.control_predicate_catalog import (
    PredicateExample,
    predicate_examples,
)
from examples.algorithms.controlled_via_and import ControlledHadamardExample
from examples.algorithms.swap_network_catalog import (
    MultiplexedSwapExample,
    SwapWithZeroExample,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    FixedTCall,
    ZeroCostCall,
)
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
    integer_basis_function_data,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.primitives.basis_state import (
    basis_effect_cost_key,
    basis_state_cost_key,
)
from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    relative_swap_signature,
)
from flagquantum.compiler.registers.primitives.swap_component import SwapDeclaration
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
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
from tools.development.notebook_autogen import NotebookSpec
from tools.development.notebook_catalog import NOTEBOOKS_BY_SECTION


def _identity(value: object) -> object:
    return value


def _example(identifier: str, value: object) -> ComponentExample[object]:
    return ComponentExample(partial(_identity, value), identifier, type(value))


_BASIC_COMPONENTS = frozenset(
    example.make() for _, _, examples in basic.DOCUMENTS for example in examples
)
_predicates = predicate_examples()
DOCUMENTS = (
    ("And", "register_and", (_example("and_bloq", ConjunctionCall((1, 1))),)),
    (
        "And",
        "register_and",
        (_example("multi_and", ConjunctionCall((1, 0, 1, 0, 1, 0))),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("zero_state", basis_state_cost_key("z", 0)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("zero_effect", basis_effect_cost_key("z", 0)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("one_state", basis_state_cost_key("z", 1)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("one_effect", basis_effect_cost_key("z", 1)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (
            _example(
                "int_state",
                integer_basis_function("IntState", opaque_register_type(8), 55),
            ),
        ),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (
            _example(
                "int_effect",
                integer_basis_function(
                    "IntEffect", opaque_register_type(8), 55, state=False
                ),
            ),
        ),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("plus_state", basis_state_cost_key("x", 0)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("plus_effect", basis_effect_cost_key("x", 0)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("minus_state", basis_state_cost_key("x", 1)),),
    ),
    (
        "States and Effects",
        "register_states_effects",
        (_example("minus_effect", basis_effect_cost_key("x", 1)),),
    ),
    (
        "Basic Swaps",
        "register_swap",
        (_example("swap_bit", CliffordCall("swap_bit", 2)),),
    ),
    ("Basic Swaps", "register_swap", (_example("cswap_bit", FixedTCall("fredkin")),)),
    (
        "Basic Swaps",
        "register_swap",
        tuple(
            _example(identifier, SwapDeclaration(width))
            for identifier, width in (
                ("swap", Symbol("n", positive=True, integer=True)),
                ("swap_small", 4),
                ("swap_large", 64),
            )
        ),
    ),
    (
        "Basic Swaps",
        "register_swap",
        tuple(
            _example(identifier, SwapDeclaration(width, controlled=True))
            for identifier, width in (
                ("cswap", Symbol("n", positive=True, integer=True)),
                ("cswap_small", 4),
                ("cswap_large", 64),
            )
        ),
    ),
    (
        "Swap Networks",
        "swap_network",
        tuple(
            _example(identifier, ControlledSwapCall(width, approximate=True))
            for identifier, width in (
                ("approx_cswap_symb", Symbol("n")),
                ("approx_cswap_small", 4),
                ("approx_cswap_large", 64),
            )
        ),
    ),
    (
        "Swap Networks",
        "swap_network",
        (
            _example("swz", SwapWithZeroExample((8,), 32, (4,))),
            _example("swz_small", SwapWithZeroExample((3,), 2, (2,))),
            _example("swz_multi_dimensional", SwapWithZeroExample((2, 2), 2, (3, 4))),
        ),
    ),
    (
        "Swap Networks",
        "swap_network",
        (_example("multiplexed_cswap", MultiplexedSwapExample()),),
    ),
    (
        "Control Specification (And)",
        "control_predicate",
        tuple(
            _example(identifier, _predicates[identifier])
            for identifier in (
                "ctrl_on_int",
                "ctrl_on_bits",
                "ctrl_on_nd_bits",
                "ctrl_on_multiple_values",
                "ctrl_on_symbolic_cv",
                "ctrl_on_symbolic_cv_multi",
            )
        ),
    ),
    (
        "Multi control bloq via single control bloq and `And` ladder",
        "controlled_via_and",
        (
            _example("controlled_via_and_ints", ControlledHadamardExample("integers")),
            _example("controlled_via_and_qbits", ControlledHadamardExample("bits")),
        ),
    ),
)


def name(component: object) -> str:
    if isinstance(component, ComponentFunction):
        return "ComponentFunction"
    if component in _BASIC_COMPONENTS:
        return basic.name(component)
    if isinstance(component, ConjunctionCall):
        return (
            "And"
            if isinstance(component.controls, tuple) and len(component.controls) == 2
            else "MultiAnd"
        )
    if isinstance(component, ZeroCostCall) and component.opcode in (
        "z_state",
        "z_effect",
        "x_state",
        "x_effect",
    ):
        prefix = (
            ("Plus", "Minus") if component.opcode.startswith("x") else ("Zero", "One")
        )
        return prefix[bool(component.parameters[0])] + (
            "State" if component.opcode.endswith("state") else "Effect"
        )
    if (
        isinstance(component, FunctionRecord)
        and integer_basis_function_data(component) is not None
    ):
        return component.name
    if isinstance(component, CliffordCall) and component.opcode == "swap_bit":
        return "TwoBitSwap"
    if isinstance(component, FixedTCall) and component.opcode == "fredkin":
        return "TwoBitCSwap"
    if isinstance(component, SwapDeclaration):
        return "CSwap" if component.controlled else "Swap"
    if isinstance(component, ControlledSwapCall) and component.approximate:
        return "CSwapApprox"
    if isinstance(component, SwapWithZeroExample):
        return "SwapWithZero"
    if isinstance(component, MultiplexedSwapExample):
        return "MultiplexedCSwap"
    if isinstance(component, PredicateExample):
        return "CtrlSpecAnd"
    if isinstance(component, ControlledHadamardExample):
        return "ControlledViaAnd"
    raise TypeError("Register-gate UI requires a registered native instance")


def _control_attributes(
    component: PredicateExample | ControlledHadamardExample,
) -> Mapping[str, object]:
    if isinstance(component, PredicateExample):
        controls, patterns = component.controls, component.patterns
    else:
        controls = tuple(
            register
            for register in component.signature
            if register.name.startswith("ctrl")
        )
        patterns = component.groups
    # Keep typed shapes, encodings and concrete activation values. The original
    # UI's unsupported-array sentinel loses these distinctions and collides.
    return {
        "controls": tuple(
            {
                "name": register.name,
                "encoding": register.dtype.encoding,
                "bitsize": register.bitsize,
                "shape": register.shape_symbolic,
                "values": None if patterns is None else patterns[index].values,
            }
            for index, register in enumerate(controls)
        ),
    }


def attributes(component: object) -> Mapping[str, object]:
    if isinstance(component, ComponentFunction):
        return component.attributes()
    if component in _BASIC_COMPONENTS:
        return basic.attributes(component)
    name(component)
    if isinstance(component, ConjunctionCall):
        if isinstance(component.controls, tuple) and len(component.controls) == 2:
            return {
                "cv1": component.controls[0],
                "cv2": component.controls[1],
                "uncompute": component.adjoint,
            }
        return {"cvs": component.controls}
    if isinstance(component, ZeroCostCall):
        return {
            "bit": bool(component.parameters[0]),
            "state": component.opcode.endswith("state"),
            "n": 1,
        }
    if isinstance(component, FunctionRecord):
        data = integer_basis_function_data(component)
        assert data is not None and component.registers is not None
        return {
            "val": data[0],
            "bitsize": component.registers[0].bitsize,
            "state": data[1],
        }
    if isinstance(component, SwapDeclaration):
        return {"bitsize": component.bitsize}
    if isinstance(component, ControlledSwapCall):
        return {"bitsize": component.width}
    if isinstance(component, SwapWithZeroExample):
        return {
            "selection_bitsizes": component.widths,
            "target_bitsize": component.word,
            "n_target_registers": component.shape,
            "uncompute": component.adjoint,
        }
    if isinstance(component, MultiplexedSwapExample):
        selection = component.layout.selections[0]
        return {
            "selection_regs": (
                {
                    "name": selection.name,
                    "dtype": {
                        "bitsize": selection.bitsize,
                        "iteration_length": dict(selection.dtype.parameters)["bound"],
                    },
                    "_shape": selection.shape_symbolic,
                    "side": selection.side,
                },
            ),
            "target_bitsize": component.word,
            "control_regs": (),
        }
    if isinstance(component, (PredicateExample, ControlledHadamardExample)):
        return _control_attributes(component)
    return {}


def filename(component: object) -> str:
    if isinstance(component, ComponentFunction):
        return component.filename()
    return component_filename(attributes(component))


def signature(component: object) -> Signature:
    if isinstance(component, ComponentFunction):
        return Signature(component.function.registers or ())
    if component in _BASIC_COMPONENTS:
        return basic.signature(component)
    name(component)
    if isinstance(component, ConjunctionCall):
        assert isinstance(component.controls, tuple)
        return and_signature(
            len(component.controls),
            multi=len(component.controls) != 2,
            erase=component.adjoint,
        )
    if isinstance(component, ZeroCostCall):
        side = (
            RegisterSide.RIGHT
            if component.opcode.endswith("state")
            else RegisterSide.LEFT
        )
        return Signature((Register("q", quantum_bit_type(), side=side),))
    if isinstance(component, FunctionRecord):
        assert component.registers is not None
        return Signature(component.registers)
    if isinstance(component, CliffordCall):
        return Signature.build(x=1, y=1)
    if isinstance(component, FixedTCall):
        return relative_swap_signature(1)
    if isinstance(component, ControlledSwapCall):
        return relative_swap_signature(component.width)
    if isinstance(
        component,
        (
            SwapDeclaration,
            SwapWithZeroExample,
            MultiplexedSwapExample,
            PredicateExample,
            ControlledHadamardExample,
        ),
    ):
        return component.signature
    raise TypeError("Register-gate UI requires a registered native instance")


def score(component: object) -> object:
    """Draw owned decompositions, falling back to the existing atomic interface."""
    if isinstance(component, ComponentFunction):
        return component.score()
    decomposed = decomposition_score(component)
    if decomposed is not None:
        return decomposed
    if component in _BASIC_COMPONENTS:
        return basic.score(component)
    ports = signature(component)
    leaf = FunctionRecord(name(component), (), registers=tuple(ports))
    builder = ComponentProgramBuilder(ports, (leaf,), name="RegisterGateInterface")
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    return register_score_data(
        component_register_score(record, "RegisterGateInterface")
    )


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    component = documentation.examples[0].make()
    if component in _BASIC_COMPONENTS:
        return basic.documentation_text(documentation)
    return (
        f"# {name(component)}\n\nNative owner `{type(component).__module__}.{type(component).__name__}`.\n\n"
        f"Register interface `{signature(component)}`.\n\n"
        "This export draws available native typed decompositions and uses the atomic "
        "typed interface for the remaining components. Execution, numerical effects, "
        "workspace and resource estimates use the existing domain owners. "
        "Large and symbolic examples are displayed without physical simulation.\n"
    )


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, NotebookSpec] = {}
    for title, stem, examples in DOCUMENTS:
        if title not in grouped:
            grouped[title] = NotebookSpec(
                title=title,
                module=sys.modules[__name__],
                component_specs=[],
                directory=str(Path(__file__).resolve().parents[1] / "notebooks"),
                path_stem=stem,
            )
        grouped[title].component_specs.append(
            ComponentDocumentation(
                component_type=examples[0].component_type,
                examples=examples,
                import_line="from examples.algorithms.register_gate_ui_export import sections",
            )
        )
    return (("Basic Gates", tuple(grouped.values())),)


def basic_gate_sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    """Use the existing catalog order for all fifteen native notebook entries."""
    by_stem = {
        spec.path_stem: spec
        for _, specs in (*basic.sections(), *sections())
        for spec in specs
    }
    return (
        (
            "Basic Gates",
            tuple(by_stem[Path(path).stem] for path in NOTEBOOKS_BY_SECTION[0][1]),
        ),
    )


def export_example(destination: Path) -> None:
    """Publish the complete Basic Gates instance catalog and fresh interfaces."""
    graph = dependency_graph(
        (
            example.make()
            for _, _, examples in (*basic.DOCUMENTS, *DOCUMENTS)
            for example in examples
        ),
        owner=lambda component: name(component) + "/" + filename(component),
    )
    export_component_ui(
        destination,
        basic_gate_sections(),
        graph,
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )


if __name__ == "__main__":
    export_example(Path("register_gate_ui_export"))
