# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 swap, predicate and fanout graphs.

"""Qualify stored child drawings against their existing native constructors."""

from dataclasses import replace

from examples.algorithms.basic_gate_decomposition import (
    decomposition_program,
    decomposition_symbols,
)
from examples.algorithms.control_predicate_catalog import PredicateExample
from examples.algorithms.controlled_via_and import ControlledHadamardExample
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlledSwapCall,
)
from flagquantum.compiler.registers.bits.fanout_component import (
    fanout_component_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.primitives.swap_component import SwapDeclaration
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.conjunction_symbols import (
    and_wire_symbol,
    multi_and_wire_symbol,
)
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.partition_symbols import split_join_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    RegisterWireSymbolSupplier,
    SymbolClass,
    adjoint_wire_symbol,
)


def _fanout_symbols(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    if step.name == "fanout_cx":
        return controlled_not_wire_symbol(step, register, index)
    return split_join_wire_symbol(step, register, index)


def _predicate(root: object) -> PredicateExample | None:
    if isinstance(root, PredicateExample):
        return root
    if isinstance(root, ControlledHadamardExample):
        return PredicateExample(
            tuple(
                Register(
                    f"ctrl_{index}",
                    register.dtype,
                    register.shape_symbolic,
                    register.side,
                )
                for index, register in enumerate(tuple(root.signature)[:-1])
            ),
            root.groups,
        )
    return None


def _predicate_symbols(
    record: ProgramRecord, component: PredicateExample, uncompute: bool
) -> RegisterWireSymbolSupplier:
    fallback = decomposition_symbols(record, component)
    activation = component.source_child().controls
    if type(activation) is int:
        activation = (1,) * activation
    assert isinstance(activation, tuple)

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if not step.name.endswith("_conjunction"):
            return fallback(step, register, index)
        original = (
            register.adjoint() if uncompute and register is not None else register
        )
        if len(activation) == 2:
            return and_wire_symbol(
                step, register, index, values=(activation[0], activation[1])
            )
        symbol = multi_and_wire_symbol(step, original, index, values=activation)
        return adjoint_wire_symbol(*symbol) if uncompute else symbol

    return supply


def child_decomposition(
    record: ProgramRecord, entry: str, root: object
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Return a qualified child graph only after literal definition matching.

    Constructor candidates follow actual native port domains and root control
    metadata. Stored helpers retain contextual identities. Full reachable bodies
    must agree, including phases, feedback and workspace, before a constructor's
    glyphs are reused. Unrecognized and primitive children retain native display.
    """
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    ports = tuple(register.name for register in registers)
    candidates: list[tuple[str, ProgramRecord, RegisterWireSymbolSupplier]] = []
    predicate = _predicate(root)
    if ports == ("ctrl", "x", "y"):
        width = registers[1].bitsize
        if type(width) is int:
            for label, component in (
                ("CSwap", SwapDeclaration(width, controlled=True)),
                ("CSwapApprox", ControlledSwapCall(width, approximate=True)),
            ):
                template = decomposition_program(component)
                assert template is not None
                candidates.append(
                    (label, template, decomposition_symbols(template, component))
                )
    elif ports == ("control", "targets"):
        width = registers[1].bitsize
        if type(width) is int:
            candidates.append(
                ("MultiTargetCNOT", fanout_component_program(width), _fanout_symbols)
            )
    elif ports == ("ctrl", "junk", "target"):
        conjunction = (
            root
            if isinstance(root, ConjunctionCall)
            else (predicate.source_child() if predicate is not None else None)
        )
        if conjunction is not None:
            if type(conjunction.controls) is int:
                conjunction = replace(conjunction, controls=(1,) * conjunction.controls)
            conjunction = replace(
                conjunction, adjoint=registers[-1].side == RegisterSide.LEFT
            )
            template = decomposition_program(conjunction)
            if template is not None:
                label = "MultiAnd†" if conjunction.adjoint else "MultiAnd"
                candidates.append(
                    (label, template, decomposition_symbols(template, conjunction))
                )
    elif predicate is not None and ports == tuple(r.name for r in predicate.signature):
        uncompute = registers[-1].side == RegisterSide.LEFT
        template = control_predicate_program(
            predicate.controls, predicate.patterns, uncompute=uncompute
        )
        candidates.append(
            (
                "CtrlSpecAnd†" if uncompute else "CtrlSpecAnd",
                template,
                _predicate_symbols(template, predicate, uncompute),
            )
        )
    return next(
        (
            candidate
            for candidate in candidates
            if matches_component_definition(record, entry, candidate[1])
        ),
        None,
    )
