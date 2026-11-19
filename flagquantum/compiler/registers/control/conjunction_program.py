# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 And and MultiAnd._decompose_via_tree.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Directional typed conjunction calls over the shared physical And emitter."""

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from .multi_control import multi_and_ir, multi_and_stages


def and_component_program(
    values: tuple[int, int], *, name: str, uncompute: bool = False
) -> ProgramRecord:
    """Produce one clean target or erase its known conjunction by measurement."""
    side = RegisterSide.LEFT if uncompute else RegisterSide.RIGHT
    bit = quantum_bit_type()
    gates = multi_and_ir(
        (0, 1),
        values,
        (2,),
        classical_bits=(0,) if uncompute else (),
        uncompute=uncompute,
    )
    leaf = bit_gate_function(
        "gates", (Register("bits", bit, (3,)),), CircuitIR(3, gates)
    )
    builder = ComponentProgramBuilder(
        Signature((Register("ctrl", bit, (2,)), Register("target", bit, side=side))),
        (leaf,),
        name=name,
    )
    target = builder.inputs["target"] if uncompute else builder.allocate()
    controls = np.asarray(builder.inputs["ctrl"])
    result = np.asarray(
        builder.call_dict(
            "gates", {"bits": np.array([*controls, target], dtype=object)}
        )["bits"]
    )
    outputs: dict[str, BindingCollection] = {"ctrl": result[:2]}
    if uncompute:
        builder.free(result[2])
    else:
        outputs["target"] = result[2]
    return _namespace_helpers(builder.finalize(outputs), name)


def conjunction_program(
    values: tuple[int, ...], *, name: str = "multi_and", uncompute: bool = False
) -> ProgramRecord:
    """Retain each And call and the live prefix ports in the source ladder."""
    stages = tuple(multi_and_stages(values))
    bit = quantum_bit_type()
    side = RegisterSide.LEFT if uncompute else RegisterSide.RIGHT
    patterns = tuple(dict.fromkeys(stage[3] for stage in stages))
    leaves = {
        pattern: and_component_program(
            pattern, name=f"and_{pattern[0]}{pattern[1]}", uncompute=uncompute
        )
        for pattern in patterns
    }
    n = len(values)
    signature = Signature(
        (
            Register("ctrl", bit, (n,)),
            *((Register("junk", bit, (n - 2,), side=side),) if n > 2 else ()),
            Register("target", bit, side=side),
        )
    )
    builder = ComponentProgramBuilder(
        signature, (f for leaf in leaves.values() for f in leaf.functions), name=name
    )
    wires = dict(enumerate(np.asarray(builder.inputs["ctrl"]).flat))
    if uncompute:
        if n > 2:
            wires.update(
                (n + i, value)
                for i, value in enumerate(np.asarray(builder.inputs["junk"]).flat)
            )
        wires[2 * n - 2] = builder.inputs["target"]
    for left, right, target, pattern in reversed(stages) if uncompute else stages:
        inputs: dict[str, BindingCollection] = {
            "ctrl": np.array([wires[left], wires[right]], dtype=object)
        }
        if uncompute:
            inputs["target"] = wires[target]
        result = builder.call_dict(leaves[pattern].functions[0].name, inputs)
        controls = np.asarray(result["ctrl"])
        wires[left], wires[right] = controls[0], controls[1]
        if not uncompute:
            output = result["target"]
            assert isinstance(output, RegisterBinding)
            wires[target] = output
    outputs = {"ctrl": np.array([wires[i] for i in range(n)], dtype=object)}
    if not uncompute:
        outputs["target"] = wires[2 * n - 2]
        if n > 2:
            outputs["junk"] = np.array(
                [wires[i] for i in range(n, 2 * n - 2)], dtype=object
            )
    return builder.finalize(outputs)
