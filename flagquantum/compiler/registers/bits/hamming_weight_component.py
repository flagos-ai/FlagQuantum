# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute and its exact adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Store the Hamming compression tree in shared typed register calls."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program
from .bit_compression_structure import compression_components
from .hamming_weight import hamming_weight_requirements
from .hamming_weight_structure import hamming_weight_stages


def hamming_weight_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Compute a sided count tree or erase a valid computed input, MSB first.

    The x word changes during compression and is restored by the adjoint.
    Forward junk and out are caller-visible owned outputs. Inverse consumes both
    after restoring every bit to zero. Each junk bit is a shared And target,
    measurement-erased in reverse source order. There is no private workspace.
    Width one retains an explicit empty opaque junk word. Invalid widths and
    nonboolean directions raise ValueError through existing validation.
    """
    junk_width, output_width = hamming_weight_requirements(width)
    if type(adjoint) is not bool:
        raise ValueError("Hamming adjoint must be boolean")
    x_dtype = RegisterType("uint", width)
    junk_dtype = RegisterType("opaque", junk_width)
    out_dtype = RegisterType("uint", output_width)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
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
    conjunction = and_component_program((1, 1), name="hamming_and", uncompute=adjoint)
    builder = ComponentProgramBuilder(signature, (cnot, *conjunction.functions))
    x = tuple(range(width))
    junk = tuple(range(width, width + junk_width))
    out = tuple(range(width + junk_width, width + junk_width + output_width))
    source = builder.inputs["x"]
    assert isinstance(source, RegisterBinding)
    wires: dict[int, RegisterBinding] = dict(zip(x, builder.split(source), strict=True))
    if adjoint:
        garbage, result = builder.inputs["junk"], builder.inputs["out"]
        assert isinstance(garbage, RegisterBinding) and isinstance(
            result, RegisterBinding
        )
        wires.update(zip(junk, builder.split(garbage), strict=True))
    else:
        result = builder.allocate(dtype=out_dtype)
    wires.update(zip(out, builder.split(result), strict=True))
    components = tuple(
        component
        for kind, addresses in hamming_weight_stages(x, junk, out)
        for component in (
            compression_components(*addresses)
            if kind == "compress"
            else (("cx", addresses),)
        )
    )
    for kind, addresses in reversed(components) if adjoint else components:
        a, b = addresses[:2]
        if kind == "cx":
            values = builder.call_dict(
                cnot.name, {"ctrl": wires[a], "target": wires[b]}
            )
            control, target = values["ctrl"], values["target"]
            assert isinstance(control, RegisterBinding) and isinstance(
                target, RegisterBinding
            )
            wires[a], wires[b] = control, target
        else:
            inputs: dict[str, BindingCollection] = {
                "ctrl": np.array([wires[a], wires[b]], dtype=object)
            }
            if adjoint:
                inputs["target"] = wires.pop(addresses[2])
            values = builder.call_dict("hamming_and", inputs)
            controls = np.asarray(values["ctrl"])
            wires[a], wires[b] = controls[0], controls[1]
            if not adjoint:
                target = values["target"]
                assert isinstance(target, RegisterBinding)
                wires[addresses[2]] = target
    outputs: dict[str, BindingCollection] = {
        "x": builder.join([wires[index] for index in x], x_dtype)
    }
    result = builder.join([wires[index] for index in out], out_dtype)
    if adjoint:
        builder.free(result)
    else:
        outputs["junk"] = builder.join(
            np.array([wires[index] for index in junk], dtype=object), junk_dtype
        )
        outputs["out"] = result
    return builder.finalize(outputs)
