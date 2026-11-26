# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 multiplexers/select_pauli_lcu.py.

"""Retain bounded unary calls and signed controlled Pauli leaf boundaries."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_expansion import expand_component_calls_once
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import WireCall, wire_component_program
from ..control.unary_wire_calls import (
    adjoint_unary_wire_program,
    append_nested_unary_wire_calls,
    bit_wire_call,
)


def pauli_select_component_pair(
    selection_width: int,
    target_width: int,
    words: tuple[str, ...],
    *,
    control_value: int | None = None,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Use MSB bounded selection and caller-owned targets, retaining signed leaves.

    A source Controlled(DensePauliString) leaf omits identity target positions.
    Its remaining QBit array is scalar only at length one. The unary owner
    owns measured predicate work; the explicit inverse reverses those calls.
    Zero-width source selection has no upstream decomposition.
    """
    if (
        type(selection_width) is not int
        or selection_width < 1
        or type(target_width) is not int
        or target_width < 0
        or not words
        or len(words) > 1 << selection_width
        or control_value is not None
        and (type(control_value) is not int or control_value not in (0, 1))
        or any(
            type(word) is not str
            or len(word.removeprefix("-")) != target_width
            or set(word.removeprefix("-")) - set("IXYZ")
            for word in words
        )
        or control_value is None
        and len(words) == 1
    ):
        raise ValueError(
            "Pauli SELECT requires finite bounded words and supported controls"
        )
    bit = quantum_bit_type()
    signature = Signature(
        (
            *((Register("control", bit),) if control_value is not None else ()),
            Register(
                "selection", bounded_unsigned_register_type(selection_width, len(words))
            ),
            Register("target", opaque_register_type(target_width)),
        )
    )
    ports: dict[str, tuple[int, ...]] = {}
    offset = 0
    for reg in signature:
        ports[reg.name] = tuple(range(offset, offset + int(reg.bitsize)))
        offset += int(reg.bitsize)
    calls: list[WireCall] = []
    leaves: list[tuple[ProgramRecord, ProgramRecord]] = []

    def leaf(index: int, enabled: int) -> None:
        word = words[index]
        active = [
            (letter, wire)
            for letter, wire in zip(
                word.removeprefix("-"), ports["target"], strict=True
            )
            if letter != "I"
        ]
        registers = (
            Register("ctrl", bit),
            *(
                (Register("q", bit, (len(active),) if len(active) != 1 else ()),)
                if active
                else ()
            ),
        )
        gates = tuple(
            Instruction("c" + letter.lower(), (0, position + 1))
            for position, (letter, _) in enumerate(active)
        )
        if word.startswith("-"):
            gates += (Instruction("z", (0,)),)
        function = bit_gate_function(
            "pauli_leaf_" + str(index), registers, CircuitIR(1 + len(active), gates)
        )
        record = ProgramRecord((function,))
        # The source controlled dense Pauli adjoint is the same Pauli leaf.
        # Preserve its decomposition order inside the reversed outer graph.
        inverse = bit_gate_function(
            function.name + "_inverse",
            registers,
            CircuitIR(1 + len(active), gates),
        )
        leaves.append((record, ProgramRecord((inverse,))))
        calls.append(
            (
                record,
                {
                    "ctrl": (enabled,),
                    **({"q": tuple(wire for _, wire in active)} if active else {}),
                },
            )
        )

    if control_value == 0:
        calls.append(bit_wire_call(Instruction("x", ports["control"])))
    append_nested_unary_wire_calls(
        calls,
        (ports["selection"],),
        (len(words),),
        tuple(range(offset, offset + selection_width)),
        leaf,
        control=ports["control"][0] if control_value is not None else None,
    )
    if control_value == 0:
        calls.append(bit_wire_call(Instruction("x", ports["control"])))
    wire_signature = Signature(
        register for register in signature if target_width or register.name != "target"
    )
    wire_ports = {
        name: wires for name, wires in ports.items() if target_width or name != "target"
    }
    forward = wire_component_program(wire_signature, wire_ports, wire_ports, calls)
    if not target_width:
        # The empty word owns a logical handle, never a physical wire identity.
        wrapper = ComponentProgramBuilder(signature)
        outputs = wrapper.call_program(
            forward, {name: wrapper.inputs[name] for name in wire_ports}
        )
        outputs["target"] = wrapper.inputs["target"]
        forward = expand_component_calls_once(wrapper.finalize(outputs))[0]
    return forward, adjoint_unary_wire_program(forward, tuple(leaves))
