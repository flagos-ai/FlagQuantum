# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 ApplyLthBloq and UnaryIterationGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/apply_lth_bloq.py

"""Call explicitly owned positive-controlled children through typed unary loops."""

from math import prod

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _collections, _elements
from .unary_component_calls import (
    append_bounded_unary_calls,
    bounded_unary_adjoint_pairs,
)


def indexed_selection_component_program(
    selections: tuple[Register, ...],
    targets: tuple[Register, ...],
    shape: tuple[int, ...],
    controlled_children: tuple[ProgramRecord, ...],
    *,
    inverses: tuple[ProgramRecord, ...] = (),
    control: Register | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain source row-major leaves and MSB-first bounded selector traversal.

    Each child exposes one scalar positive control followed by target elements
    in the source positional .on(control, *targets) order. All target storage
    remains caller-owned. The unary owner creates and measurement-erases private
    predicates. An existing source control is positive even for the upstream
    solitary-negative control-value defect; its activation metadata is handled
    by the source specialization owner, not inferred here.
    """
    if type(adjoint) is not bool or len(shape) != len(selections) or not shape:
        raise ValueError(
            "Indexed selection requires matching axes and boolean direction"
        )
    if any(type(size) is not int or size < 1 for size in shape) or prod(shape) < 2:
        raise ValueError("Indexed selection requires at least two finite children")
    if len(controlled_children) != prod(shape) or (
        adjoint and len(inverses) != len(controlled_children)
    ):
        raise ValueError("Indexed selection requires complete directional child owners")
    if any(
        reg.shape
        or type(reg.bitsize) is not int
        or reg.bitsize < 1
        or size > 2**reg.bitsize
        for reg, size in zip(selections, shape, strict=True)
    ):
        raise ValueError("Indexed selection requires finite scalar selector words")
    if control is not None and (control.shape or control.total_qubits() != 1):
        raise ValueError("Indexed selection control must be one scalar qubit")
    widths = tuple(reg.bitsize for reg in targets for _ in range(prod(reg.shape)))
    ports: list[tuple[Register, ...]] = []
    for child in controlled_children:
        registers = child.functions[0].registers
        if (
            registers is None
            or not Signature(registers).thru_registers_only
            or not registers
        ):
            raise ValueError("Indexed child requires typed through ports")
        if registers[0].shape or registers[0].total_qubits() != 1:
            raise ValueError("Indexed child requires a leading scalar positive control")
        if (
            tuple(reg.bitsize for reg in registers[1:] for _ in range(prod(reg.shape)))
            != widths
        ):
            raise ValueError(
                "Indexed child target elements must match the source layout"
            )
        ports.append(registers)
    signature = Signature(
        (*((control,) if control is not None else ()), *selections, *targets)
    )
    builder = ComponentProgramBuilder(signature)
    state: dict[str, BindingCollection] = {
        reg.name: builder.inputs[reg.name] for reg in targets
    }
    selectors: dict[int, tuple[RegisterBinding, ...]] = {}

    def visit(
        axis: int, row: int, active: RegisterBinding | None
    ) -> RegisterBinding | None:
        if axis not in selectors:
            word = builder.inputs[selections[axis].name]
            assert isinstance(word, RegisterBinding)
            selectors[axis] = (
                (word,) if selections[axis].bitsize == 1 else tuple(builder.split(word))
            )
        stride = prod(shape[axis + 1 :])

        def leaf(index: int, predicate: RegisterBinding) -> RegisterBinding:
            nonlocal state
            selected = row + index * stride
            if axis + 1 < len(shape):
                successor = visit(axis + 1, selected, predicate)
                assert successor is not None
                return successor
            child = controlled_children[selected]
            child_ports = ports[selected]
            values = _collections(child_ports[1:], _elements(state.values()))
            result = builder.call_program(
                child,
                {child_ports[0].name: predicate, **values},
                entry=child.functions[0].name,
            )
            returned_control = result.pop(child_ports[0].name)
            assert isinstance(returned_control, RegisterBinding)
            state = _collections(targets, _elements(result.values()))
            return returned_control

        selectors[axis], active = append_bounded_unary_calls(
            builder, selectors[axis], shape[axis], leaf, control=active
        )
        return active

    active = builder.inputs[control.name] if control is not None else None
    assert active is None or isinstance(active, RegisterBinding)
    active = visit(0, 0, active)
    outputs = dict(state)
    for axis, register in enumerate(selections):
        outputs[register.name] = (
            selectors[axis][0]
            if register.bitsize == 1
            else builder.join(selectors[axis], register.dtype)
        )
    if control is not None:
        assert active is not None
        outputs[control.name] = active
    record = builder.finalize(outputs)
    return (
        adjoint_component_calls(
            record,
            (
                *zip(controlled_children, inverses, strict=True),
                *bounded_unary_adjoint_pairs(),
            ),
        )
        if adjoint
        else record
    )
