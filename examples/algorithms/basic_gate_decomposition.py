# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 swap.py, and_bloq.py and ctrl_spec_and.py.

"""Draw existing typed swap and predicate decompositions with source glyphs.

Compiler Programs retain all call identities and register lifetimes. This
presentation adapter selects their immediate graph and supplies wire symbols.
It neither expands physical instructions nor evaluates resource estimates.
"""

from examples.algorithms.control_predicate_catalog import PredicateExample
from examples.algorithms.swap_network_catalog import SwapWithZeroExample
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlledSwapCall,
)
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function_data,
    integer_basis_program,
)
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    conjunction_program,
)
from flagquantum.compiler.registers.lookup.swap_with_zero_component import (
    swap_with_zero_program,
)
from flagquantum.compiler.registers.primitives.relative_swap_component import (
    relative_swap_program,
)
from flagquantum.compiler.registers.primitives.swap_component import SwapDeclaration
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def decomposition_program(component: object) -> ProgramRecord | None:
    """Select only complete concrete graphs owned by the existing Compiler."""
    if isinstance(component, SwapWithZeroExample):
        return swap_with_zero_program(
            component.widths, component.word, component.shape, adjoint=component.adjoint
        )
    if isinstance(component, FunctionRecord):
        data = integer_basis_function_data(component)
        if data is not None:
            assert component.registers is not None
            return integer_basis_program(
                component.name, component.registers[0].dtype, data[0], state=data[1]
            )
    if (
        isinstance(component, ControlledSwapCall)
        and component.approximate
        and not component.adjoint
        and type(component.width) is int
    ):
        return relative_swap_program(component.width)
    if isinstance(component, SwapDeclaration) and type(component.bitsize) is int:
        return component.program()
    if (
        isinstance(component, ConjunctionCall)
        and isinstance(component.controls, tuple)
        and len(component.controls) > 2
        and all(type(value) is int for value in component.controls)
    ):
        return conjunction_program(component.controls, uncompute=component.adjoint)
    if isinstance(component, PredicateExample) and type(component.width) is int:
        return control_predicate_program(component.controls, component.patterns)
    return None


def _box(
    register: Register, text: str
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    glyphs: dict[RegisterSide, SymbolClass] = {
        RegisterSide.LEFT: "RarrowTextBox",
        RegisterSide.RIGHT: "LarrowTextBox",
        RegisterSide.THRU: "TextBox",
    }
    return glyphs[register.side], {"text": text}


def decomposition_symbols(
    record: ProgramRecord, component: object
) -> RegisterWireSymbolSupplier:
    """Use native child definitions and original activation metadata for glyphs."""
    functions = {function.name: function for function in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            if isinstance(component, SwapWithZeroExample) and tuple(
                r.name for r in functions[step.name].registers or ()
            ) == ("ctrl", "x", "y"):
                return "Text", {"text": "~swap", "fontsize": 10}
            if tuple(r.name for r in functions[step.name].registers or ()) == (
                "control",
                "targets",
            ):
                return "Text", {"text": "MultiTargetCNOT", "fontsize": 10}
            return "Text", {
                "text": (
                    "TwoBitSwap"
                    if step.name == "swap_bit"
                    and len(functions[step.name].registers or ()) == 2
                    else ""
                ),
                "fontsize": 10,
            }
        assert index is not None
        function = functions[step.name]
        if isinstance(component, SwapWithZeroExample) and register.name in {
            "ctrl",
            "x",
            "y",
        }:
            return (
                ("TextBox", {"text": "@(approx)"})
                if register.name == "ctrl"
                else ("TextBox", {"text": f"×({register.name})"})
            )
        if isinstance(component, ControlledSwapCall):
            if step.name in {"s", "sdg", "h", "t", "tdg"}:
                return "TextBox", {
                    "text": {"s": "S", "sdg": "S†", "h": "H", "t": "T", "tdg": "T†"}[
                        step.name
                    ]
                }
            if step.name == "cx":
                return (
                    ("Circle", {"filled": True})
                    if register.name == "ctrl"
                    else ("ModPlus", {})
                )
            if register.name == "control":
                return "Circle", {"filled": True}
            if register.name == "targets":
                return "ModPlus", {}
        if isinstance(component, FunctionRecord) and step.name.endswith(
            ("_basis_0", "_basis_1")
        ):
            return _box(register, step.name[-1])
        if step.name == "swap_bit":
            if register.name == "ctrl":
                return "Circle", {"filled": True}
            return "TextBox", {
                "text": "×" if len(function.registers or ()) == 3 else register.name
            }
        if isinstance(component, ConjunctionCall):
            # The owning conjunction Program names leaves by their two polarities.
            values = tuple(map(int, function.name.removeprefix("and_")))
            if register.name == "ctrl":
                return "Circle", {"filled": bool(values[index[0]])}
            return _box(register, "∧")
        if isinstance(component, PredicateExample) and step.name.endswith(
            "_conjunction"
        ):
            activation = component.source_child().controls
            if not isinstance(activation, tuple):
                assert type(component.width) is int
                activation = (1,) * component.width
            if register.name == "ctrl":
                value = activation[index[0]]
                return "Circle", {
                    "filled": bool(value) if len(activation) == 2 else bool(value == 1)
                }
            text = "∧" if register.name == "target" else register.name
            if index:
                text += "[" + ", ".join(map(str, index)) + "]"
            return _box(register, text)
        operations = tuple(op for block in function.blocks for op in block.operations)
        if operations and all(
            op.opcode in {"split_register", "join_register"} for op in operations
        ):
            text = "[" + ", ".join(map(str, index)) + "]" if register.shape else " "
            return _box(register, text)
        # Partition uses its declared register labels, including shaped indices.
        text = register.name
        if index:
            text += "[" + ", ".join(map(str, index)) + "]"
        return _box(register, text)

    return supply


def decomposition_score(component: object) -> MusicalScoreData | None:
    """Project the immediate native call graph, returning None for other owners."""
    record = decomposition_program(component)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=decomposition_symbols(record, component),
    )
