# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiplexedCSwap and UnaryIterationGate.

"""Retain bounded unary traversal and exact word-exchange child calls."""

from itertools import product
from math import prod

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program, conjunction_program
from ..control.unary_component_calls import append_bounded_unary_calls
from ..primitives.controlled_swap_component import controlled_swap_program
from .multiplexed_swap import MultiplexedSwapLayout


def multiplexed_swap_program(
    layout: MultiplexedSwapLayout, *, adjoint: bool = False
) -> ProgramRecord:
    """Thread MSB-first selectors, row-major targets and caller-owned output.

    The existing layout validates concrete dimensions and source refusals. Unary
    iteration owns predicate creation and measured erasure. Selector words split
    on first use, retaining the source's nested import order. Final joins preserve
    native bounded descriptors, where the original returns generic QAny storage.
    """
    if type(adjoint) is not bool:
        raise ValueError("Multiplexed swap direction must be boolean")
    shape, _, count, word = layout._concrete_dimensions()
    builder = ComponentProgramBuilder(layout.signature)
    selectors: dict[int, tuple[RegisterBinding, ...]] = {}
    targets = np.asarray(builder.inputs["targets"]).reshape(-1).copy()
    output = builder.inputs["output"]
    child = controlled_swap_program(word)
    control_bits: list[RegisterBinding] = []
    for register in layout.controls:
        values = np.asarray(builder.inputs[register.name])
        for index in register.all_idxs():
            value = values[index]
            assert isinstance(value, RegisterBinding)
            bits = (value,) if register.bitsize == 1 else tuple(builder.split(value))
            control_bits.extend(bits)
    reduction: dict[str, BindingCollection] = {}
    predicate: RegisterBinding | None
    if count > 1:
        program = (
            and_component_program((1, 1), name="control_and")
            if count == 2
            else conjunction_program((1,) * count, name="control_and")
        )
        reduction = builder.call_program(
            program, {"ctrl": np.array(control_bits, dtype=object)}, entry="control_and"
        )
        target = reduction["target"]
        assert isinstance(target, RegisterBinding)
        predicate = target
    else:
        predicate = next(iter(control_bits), None)

    def visit(
        axis: int, row: int, control: RegisterBinding | None
    ) -> RegisterBinding | None:
        nonlocal output
        if axis not in selectors:
            value = builder.inputs[layout.selections[axis].name]
            assert isinstance(value, RegisterBinding)
            selectors[axis] = (
                (value,)
                if layout.selections[axis].bitsize == 1
                else tuple(builder.split(value))
            )
        stride = prod(shape[axis + 1 :])

        def leaf(index: int, active: RegisterBinding) -> RegisterBinding:
            nonlocal output
            selected = row + index * stride
            successor: RegisterBinding | None
            if axis + 1 < len(shape):
                successor = visit(axis + 1, selected, active)
            else:
                result = builder.call_program(
                    child, {"ctrl": active, "x": targets[selected], "y": output}
                )
                targets[selected], output = result["x"], result["y"]
                target = result["ctrl"]
                assert isinstance(target, RegisterBinding)
                successor = target
            assert isinstance(successor, RegisterBinding)
            return successor

        selectors[axis], control = append_bounded_unary_calls(
            builder, selectors[axis], shape[axis], leaf, control=control
        )
        return control

    predicate = visit(0, 0, predicate)
    if count > 1:
        assert predicate is not None
        reduction["target"] = predicate
        inverse = (
            and_component_program((1, 1), name="control_erase", uncompute=True)
            if count == 2
            else conjunction_program((1,) * count, name="control_erase", uncompute=True)
        )
        result = builder.call_program(inverse, reduction, entry="control_erase")
        control_bits = list(np.asarray(result["ctrl"]).flat)
    elif predicate is not None:
        control_bits = [predicate]
    outputs: dict[str, BindingCollection] = {}
    offset = 0
    for register in layout.controls:
        values = np.empty(register.shape, dtype=object)
        width = int(register.bitsize)
        for index in register.all_idxs():
            returned_bits = control_bits[offset : offset + width]
            values[index] = (
                returned_bits[0]
                if register.bitsize == 1
                else builder.join(returned_bits, register.dtype)
            )
            offset += width
        outputs[register.name] = values if register.shape else values[()]
    for axis, register in enumerate(layout.selections):
        outputs[register.name] = (
            selectors[axis][0]
            if register.bitsize == 1
            else builder.join(selectors[axis], register.dtype)
        )
    record = builder.finalize(
        {**outputs, "targets": targets.reshape(shape), "output": output}
    )
    if not adjoint:
        return record
    pairs = [(child, child)]
    for left, right in product((0, 1), repeat=2):
        control_values = (left, right)
        compute = and_component_program(control_values, name="compute")
        erase = and_component_program(control_values, name="erase", uncompute=True)
        pairs.extend(((compute, erase), (erase, compute)))
    if count > 2:
        compute = conjunction_program((1,) * count, name="compute")
        erase = conjunction_program((1,) * count, name="erase", uncompute=True)
        pairs.extend(((compute, erase), (erase, compute)))
    for gate, ports in (("x", ("q",)), ("cx", ("ctrl", "target"))):
        function = bit_gate_function(
            gate,
            tuple(Register(port, quantum_bit_type()) for port in ports),
            CircuitIR(len(ports), (Instruction(gate, tuple(range(len(ports)))),)),
        )
        primitive = ProgramRecord((function,))
        pairs.append((primitive, primitive))
    return adjoint_component_calls(record, tuple(pairs))
