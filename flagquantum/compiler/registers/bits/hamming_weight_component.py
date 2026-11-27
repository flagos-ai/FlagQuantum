# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute and its exact adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Store the Hamming compression tree in shared typed register calls."""

from dataclasses import replace

import numpy as np
import sympy

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_rebindings import join_word_function, split_word_function
from ..control.conjunction_program import and_component_program
from ..layout.component_cast import cast_word_function
from ..layout.component_word_views import gather_word_binding
from ..layout.component_workspace import free_word_function
from .bit_compression_structure import compression_components
from .hamming_weight import hamming_weight_requirements
from .hamming_weight_structure import hamming_weight_stages


def hamming_weight_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Compute a sided count tree or erase a valid computed input, MSB first.

    The x word changes during compression and is restored by the adjoint.
    Forward junk and out are caller-visible owned outputs. Missing output
    bits are allocated on first use, following source view gathering. Inverse consumes both
    after restoring every bit to zero. Each junk bit is a shared And target,
    measurement-erased in reverse source order. There is no private workspace.
    Width one retains an explicit empty opaque junk word. Invalid widths and
    nonboolean directions raise ValueError through existing validation.
    """
    junk_width, _ = hamming_weight_requirements(width)
    return _hamming_tree_program(width, junk_width, adjoint)


def source_hamming_weight_program(
    width: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain the source SDK's excess junk for concrete symbolic integers.

    SymPy integer literals use the source lower bound of one set bit. Their
    trailing unused junk is still an owned zero output, consumed by the inverse.
    Nonliteral symbolic widths have no finite physical program.
    """
    if type(width) is int:
        return hamming_weight_program(width, adjoint=adjoint)
    if not isinstance(width, sympy.Integer) or width < 1:
        raise ValueError("Hamming SDK execution requires a positive integer literal")
    return _hamming_tree_program(int(width), int(width) - 1, adjoint)


def _hamming_tree_program(width: int, junk_width: int, adjoint: bool) -> ProgramRecord:
    _, output_width = hamming_weight_requirements(width)
    if type(adjoint) is not bool:
        raise ValueError("Hamming adjoint must be boolean")
    if adjoint:
        return _hamming_adjoint(_hamming_tree_program(width, junk_width, False))
    x_dtype = RegisterType("uint", width)
    junk_dtype = RegisterType("opaque", junk_width)
    out_dtype = RegisterType("uint", output_width)
    side = RegisterSide.RIGHT
    signature = Signature(
        (
            Register("x", x_dtype),
            Register("junk", junk_dtype, side=side),
            Register("out", out_dtype, side=side),
        )
    )
    bit = quantum_bit_type()
    cnot = bit_gate_function(
        "hamming_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    conjunction = and_component_program((1, 1), name="hamming_and", uncompute=False)
    builder = ComponentProgramBuilder(signature, (cnot, *conjunction.functions))
    x = tuple(range(width))
    junk = tuple(range(width, width + junk_width))
    out = tuple(range(width + junk_width, width + junk_width + output_width))
    positions = {"x": x, "junk": junk, "out": out}
    words: dict[tuple[int, ...], RegisterBinding] = {}
    for register in signature.lefts():
        value = builder.inputs[register.name]
        assert isinstance(value, RegisterBinding)
        words[positions[register.name]] = value

    def bit_at(wire: int) -> RegisterBinding:
        if not any(wire in word for word in words):
            words[(wire,)] = builder.allocate(dtype=bit)
        return gather_word_binding(builder, words, (wire,), bit)

    components = tuple(
        component
        for kind, addresses in hamming_weight_stages(x, junk, out)
        for component in (
            compression_components(*addresses)
            if kind == "compress"
            else (("cx", addresses),)
        )
    )
    for kind, addresses in components:
        a, b = addresses[:2]
        control, target = bit_at(a), bit_at(b)
        if kind == "cx":
            values = builder.call_dict(cnot.name, {"ctrl": control, "target": target})
            returned_control, returned_target = values["ctrl"], values["target"]
            assert isinstance(returned_control, RegisterBinding) and isinstance(
                returned_target, RegisterBinding
            )
            words[(a,)], words[(b,)] = returned_control, returned_target
        else:
            inputs: dict[str, BindingCollection] = {
                "ctrl": np.array([control, target], dtype=object)
            }
            values = builder.call_dict("hamming_and", inputs)
            controls = np.asarray(values["ctrl"])
            words[(a,)], words[(b,)] = controls[0], controls[1]
            result = values["target"]
            assert isinstance(result, RegisterBinding)
            words[(addresses[2],)] = result
    outputs: dict[str, RegisterBinding] = {}
    for register in signature.rights():
        wires = positions[register.name]
        for wire in wires:
            if not any(wire in word for word in words):
                bit_at(wire)
        outputs[register.name] = (
            gather_word_binding(builder, words, wires, register.dtype)
            if wires
            else builder.join([], register.dtype)
        )
    retained = set(outputs.values())
    for value in words.values():
        if value not in retained:
            builder.free(value)
    return builder.finalize(outputs)


def _hamming_adjoint(record: ProgramRecord) -> ProgramRecord:
    """Reverse the source graph with the specialized measurement-erased And."""
    inverse = and_component_program((1, 1), name="hamming_erase", uncompute=True)
    definitions = {function.name: function for function in record.functions}
    definitions.update((function.name, function) for function in inverse.functions)
    mapping = {"hamming_cx": "hamming_cx", "hamming_and": "hamming_erase"}
    used = {op.callee for op in record.functions[0].blocks[0].operations}
    for function in record.functions[1:]:
        if function.name not in used or function.name in mapping:
            continue
        ports = function.registers
        assert ports is not None
        name = function.name + "_inverse"
        if len(ports) == 1:
            target = free_word_function(name, ports[0].dtype, False)
        elif ports[0].shape_symbolic:
            target = split_word_function(name, ports[1].dtype)
        elif ports[1].shape_symbolic:
            target = join_word_function(name, ports[0].dtype)
        else:
            target = cast_word_function(name, ports[1].dtype, ports[0].dtype)
        definitions[name] = target
        mapping[function.name] = name
    result = adjoint_component_program(
        replace(record, functions=tuple(definitions.values())), mapping
    )
    # Retain the existing named And boundary for call-count consumers.
    functions = []
    for function in result.functions:
        if function.name == "hamming_and":
            continue
        functions.append(
            replace(
                function,
                name=(
                    "hamming_and" if function.name == "hamming_erase" else function.name
                ),
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(op, callee="hamming_and")
                                if op.callee == "hamming_erase"
                                else op
                            )
                            for op in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
        )
    return replace(result, functions=tuple(functions))
