# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 musical_score._binst_assign_line.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Project a selected Core component graph onto the existing register score."""

from collections.abc import Callable, Iterable, Iterator, Sequence

from ..compiler.registers.component_graph import _call_entry
from ..core.ir.program import FunctionRecord, ProgramRecord, ValueRef
from ..core.ir.program.registers import Register, RegisterSide
from ..core.ir.program.signatures import Signature
from .register_lanes import RegisterLaneManager
from .register_score import RegisterScore, RegisterStep, layout_register_score


def _ports(
    registers: Iterable[Register], values: Sequence[ValueRef]
) -> Iterator[tuple[Register, tuple[int, ...], ValueRef]]:
    identities = iter(values)
    for register in registers:
        for index in register.all_idxs():
            yield register, index, next(identities)


def _label(name: str, index: tuple[int, ...]) -> str:
    return name + ("[" + ",".join(map(str, index)) + "]" if index else "")


def component_register_score(
    record: ProgramRecord,
    entry: str = "main",
    *,
    style: Callable[[FunctionRecord, RegisterStep], RegisterStep] | None = None,
    lane_manager: RegisterLaneManager | None = None,
) -> RegisterScore:
    """Display an already selected typed-call graph without executing its bodies.

    Compiler owns validation and graph selection. Stored call order seeds the
    existing layout dependency generations. This projection only names
    register lifetimes: a THRU port keeps its predecessor lane; RIGHT ports open
    fresh lanes before LEFT ports release theirs. Array elements retain row-major
    order. Existing layout, HTML and Matplotlib renderers consume the result.
    Callers may style a step but must preserve its input/output identities.
    A caller-owned lane_manager receives native register metadata and original
    call ordinals, retaining one predicate check per fresh register and one hook
    per allocated or released element. Boundary locations are input and output.

    No decomposition, primitive execution, tensor supply or backend selection
    occurs. Finite one-block logical graphs with quantum and classical words are
    admitted by the existing component traversal boundary. Zero-width scalar
    words retain a logical lane, and empty arrays have no elements. Symbolic
    words and conditioned or parameterized calls are refused.
    """
    root = _call_entry(record, entry)
    assert root.registers is not None
    block = root.blocks[0]
    functions = {function.name: function for function in record.functions}
    known: dict[ValueRef, str] = {}
    widths: dict[str, int] = {}
    inputs = []
    for register, index, value in _ports(
        Signature(root.registers).lefts(), block.arguments
    ):
        name = _label(register.name, index)
        known[value] = name
        widths[name] = int(register.bitsize)
        inputs.append(name)
    live = set(inputs)
    steps = []
    for position, operation in enumerate(block.operations):
        if operation.parameters or operation.condition_clauses:
            raise ValueError(
                "component scores require unconditional unparameterized calls"
            )
        function = functions[operation.callee or ""]
        assert function.registers is not None
        signature = Signature(function.registers)
        left = tuple(_ports(signature.lefts(), operation.operands))
        input_names = tuple(known[value] for _, _, value in left)
        through = {
            (register.name, index): known[value]
            for register, index, value in left
            if register.side == RegisterSide.THRU
        }
        output_names = []
        for register, index, value in _ports(signature.rights(), operation.results):
            key = (register.name, index)
            if register.side == RegisterSide.THRU:
                name = through[key]
            else:
                name = f"call{position}:" + _label(register.name, index)
                while name in widths:
                    name = "_" + name
                widths[name] = int(register.bitsize)
            known[value] = name
            output_names.append(name)
        step = RegisterStep(
            function.name,
            input_names,
            tuple(output_names),
            input_registers=tuple(signature.lefts()),
            output_registers=tuple(signature.rights()),
        )
        if style is not None:
            styled = style(function, step)
            if styled.inputs != step.inputs or styled.outputs != step.outputs:
                raise ValueError("component score styling cannot change connectivity")
            if (
                styled.input_registers != step.input_registers
                or styled.output_registers != step.output_registers
            ):
                raise ValueError(
                    "component score styling cannot change register metadata"
                )
            step = styled
        steps.append(step)
        live.difference_update(input_names)
        live.update(output_names)
    returned = tuple(known[value] for value in block.return_values or ())
    if len(set(returned)) != len(returned) or set(returned) != live:
        raise ValueError(
            "component score outputs must retain every live register exactly once"
        )
    signature = Signature(root.registers)
    return layout_register_score(
        widths,
        inputs,
        steps,
        lane_manager=lane_manager,
        input_registers=tuple(signature.lefts()),
        output_registers=tuple(signature.rights()),
        outputs=returned,
    )
