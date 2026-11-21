# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanConstant.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Retain the source prefix-equality scan and its measured reverse cleanup."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program
from ._component_gates import comparison_bit_functions
from .less_than import less_than_constant_workspace_size


def less_than_constant_program(width: int, threshold: int) -> ProgramRecord:
    """XOR unsigned x<threshold into a dirty target with source And history.

    The equality seed and prefix flags begin clean. Recorded inverse calls erase
    every flag. The out-of-range upper threshold requires just one X gate.
    """
    less_than_constant_workspace_size(width, threshold)
    signature = Signature.build_from_dtypes(
        x=unsigned_register_type(width), target=quantum_bit_type()
    )
    ports = {"x": tuple(range(width)), "target": (width,)}
    x, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    if threshold >= 1 << width:
        return wire_component_program(signature, ports, ports, ((x, {"q": (width,)}),))
    compute = and_component_program((1, 1), name="prefix_and")
    erase = and_component_program((1, 1), name="prefix_erase", uncompute=True)
    equal = width + 1
    calls: list[WireCall] = [(x, {"q": (equal,)})]
    cleanup = list(calls)
    pattern = tuple((threshold >> shift) & 1 for shift in reversed(range(width)))
    for index, value in enumerate(pattern):
        if value:
            flip: WireCall = x, {"q": (index,)}
            calls.append(flip)
            cleanup.append(flip)
        flag = width + 2 + index
        operands = {"ctrl": (index, equal), "target": (flag,)}
        calls.append((compute, operands))
        cleanup.append((erase, operands))
        if value:
            calls.append((cx, {"ctrl": (flag,), "target": (width,)}))
        copy: WireCall = cx, {"ctrl": (flag,), "target": (equal,)}
        calls.append(copy)
        cleanup.append(copy)
    calls.extend(reversed(cleanup))
    return wire_component_program(signature, ports, ports, calls)
