# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 prepare_uniform_superposition.py.

"""Retain uniform-state calls, typed word views and directional predicate work."""

from math import acos

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, bit_length
from ..comparison.constant_component import less_than_constant_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_via_predicate
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program, conjunction_program
from ..control.control_predicate import ControlRegister
from ..primitives.controlled_hadamard import controlled_hadamard_ir
from .uniform_preparation import _odd_factor


def _initial_hadamard(values: tuple[int, ...]) -> ProgramRecord:
    bit = quantum_bit_type()
    h = bit_gate_function(
        "uniform_h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    if not values:
        return ProgramRecord((h,))
    ch = bit_gate_function(
        "uniform_ch",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, controlled_hadamard_ir((0,), 1, control_values=(1,))),
    )
    if values == (1,):
        return ProgramRecord((ch,))
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(q=bit), (h, ch))
    returned = builder.call_dict(h.name, builder.inputs)

    def controlled(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = builder.call_dict(ch.name, {"ctrl": control, "target": values["q"]})
        return result["ctrl"], (result["target"],)

    return control_component_via_predicate(
        builder.finalize(returned),
        {"main": controlled},
        (Register("ctrl", bit, () if len(values) == 1 else (len(values),)),),
        (ControlRegister(1, values),),
    )


def uniform_preparation_program(
    count: SymbolicInt, values: tuple[int, ...] = (), *, adjoint: bool = False
) -> ProgramRecord:
    """Retain the original complete uniform recipe on arbitrary through storage.

    The opaque target is MSB-first. Odd-factor comparison scratch and the zero
    predicate are private clean work and are measurement-erased. Unknown count
    remains a declaration. The original singleton has no target body.
    """
    if type(adjoint) is not bool:
        raise ValueError("Uniform preparation direction must be boolean")
    if any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("Uniform preparation requires binary control values")
    width = bit_length(count - 1)
    bit = quantum_bit_type()
    signature = Signature.build(ctrl=len(values), target=width)
    if type(count) is not int:
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    odd = _odd_factor(count)
    if count == 1:
        raise ValueError("Original singleton uniform preparation has no target body")
    width = (count - 1).bit_length()
    control = tuple(range(len(values)))
    target = tuple(range(len(values), len(values) + width))
    ports = {**({"ctrl": control} if values else {}), "target": target}
    h = _initial_hadamard(())
    initial = _initial_hadamard(values)
    calls: list[WireCall] = []
    inverses: dict[int, ProgramRecord] = {}
    for wire in target:
        operands = (
            {"ctrl": control, "target": (wire,)}
            if values == (1,)
            else {**({"ctrl": control} if values else {}), "q": (wire,)}
        )
        calls.append((initial, operands))
    if odd != 1:
        high_width = (odd - 1).bit_length()
        high = target[:high_width]
        flag = len(values) + width
        angle = acos(1 - (1 << (high_width - 1)) / odd)
        rz = ProgramRecord(
            (
                bit_gate_function(
                    "uniform_rz",
                    (Register("q", bit),),
                    CircuitIR(1, (Instruction("rz", (0,), params={"theta": angle}),)),
                ),
            )
        )
        inverses[id(rz)] = ProgramRecord(
            (
                bit_gate_function(
                    "uniform_rz_inverse",
                    (Register("q", bit),),
                    CircuitIR(1, (Instruction("rz", (0,), params={"theta": -angle}),)),
                ),
            )
        )
        compare = less_than_constant_program(high_width, odd)
        if adjoint:
            inverses[id(compare)] = less_than_constant_program(
                high_width, odd, adjoint=True
            )
        compare_ports = {"x": high, "target": (flag,)}
        calls.extend(
            ((compare, compare_ports), (rz, {"q": (flag,)}), (compare, compare_ports))
        )
        calls.extend((h, {"q": (wire,)}) for wire in high)
        activations = (0,) * high_width + values
        conjunction = (
            and_component_program(activations, name="uniform_zero")
            if len(activations) == 2
            else conjunction_program(activations, name="uniform_zero")
        )
        erase = (
            and_component_program(
                activations, name="uniform_zero_erase", uncompute=True
            )
            if len(activations) == 2
            else conjunction_program(
                activations, name="uniform_zero_erase", uncompute=True
            )
        )
        inverses[id(conjunction)] = erase
        inverses[id(erase)] = conjunction
        # Logical identities distinguish the freed comparison flag from later
        # predicate outputs. Physical scheduling owns storage reuse.
        predicate = flag + len(activations) - 1
        operands = {"ctrl": (*high, *control), "target": (predicate,)}
        if len(activations) > 2:
            operands["junk"] = tuple(range(flag + 1, predicate))
        calls.extend(
            ((conjunction, operands), (rz, {"q": (predicate,)}), (erase, operands))
        )
        calls.extend((h, {"q": (wire,)}) for wire in high)
    record = wire_component_program(signature, ports, ports, calls)
    if not adjoint:
        return record
    components = tuple((child, inverses.get(id(child), child)) for child, _ in calls)
    return adjoint_component_calls(record, components)
