# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepSix.

"""Retain the final coordinate corrections and original flag-release promises."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def addition_finish_program(
    signature: Signature,
    *,
    zero_flags: ProgramRecord,
    zero_three_words: ProgramRecord,
    toffoli: ProgramRecord,
    zero: ProgramRecord,
    controlled_subtract: ProgramRecord,
    controlled_add: ProgramRecord,
    equals_pair: ProgramRecord,
    zero_pair: ProgramRecord,
    conjunction: ProgramRecord,
    fanout: ProgramRecord,
    conjunction_adjoint: ProgramRecord,
) -> ProgramRecord:
    """Consume five LEFT flags on the source promise; preserve four MSB words."""
    builder = ComponentProgramBuilder(signature)
    flags, triple, ccx, initialize, sub, plus, equals, pair, both, copy, both_adj = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (
            zero_flags,
            zero_three_words,
            toffoli,
            zero,
            controlled_subtract,
            controlled_add,
            equals_pair,
            zero_pair,
            conjunction,
            fanout,
            conjunction_adjoint,
        )
    )
    f1, f2, f3, f4, ctrl, a, b, x, y = (
        builder.inputs[name]
        for name in ("f1", "f2", "f3", "f4", "ctrl", "a", "b", "x", "y")
    )
    dtype = signature.get_left("a").dtype
    controls, ctrl = builder.call_tuple(
        flags, {"controls": np.array([f2, f3, f4]), "target": ctrl}
    )
    assert isinstance(controls, np.ndarray)
    f2, f3, f4 = controls
    words, f1 = builder.call_tuple(triple, {"ctrl": np.array([a, x, y]), "q": f1})
    assert isinstance(words, np.ndarray)
    a, x, y = words
    words, f1 = builder.call_tuple(triple, {"ctrl": np.array([x, a, b]), "q": f1})
    assert isinstance(words, np.ndarray)
    x, a, b = words
    words, f2 = builder.call_tuple(triple, {"ctrl": np.array([a, b, y]), "q": f2})
    assert isinstance(words, np.ndarray)
    a, b, y = words
    words, f2 = builder.call_tuple(triple, {"ctrl": np.array([x, y, b]), "q": f2})
    assert isinstance(words, np.ndarray)
    x, y, b = words
    assert isinstance(a, RegisterBinding) and isinstance(x, RegisterBinding)
    a_bits, x_bits = builder.split(a), builder.split(x)
    for i in range(len(a_bits)):
        controls, x_bits[i] = builder.call_tuple(
            ccx, {"ctrl": np.array([f4, a_bits[i]]), "target": x_bits[i]}
        )
        assert isinstance(controls, np.ndarray)
        f4, a_bits[i] = controls
    a, x = builder.join(a_bits, dtype), builder.join(x_bits, dtype)
    assert isinstance(b, RegisterBinding) and isinstance(y, RegisterBinding)
    b_bits, y_bits = builder.split(b), builder.split(y)
    for i in range(len(b_bits)):
        controls, y_bits[i] = builder.call_tuple(
            ccx, {"ctrl": np.array([f4, b_bits[i]]), "target": y_bits[i]}
        )
        assert isinstance(controls, np.ndarray)
        f4, b_bits[i] = controls
    b, y = builder.join(b_bits, dtype), builder.join(y_bits, dtype)
    ancilla = builder.call_dict(initialize, {})["q"]
    controls, ancilla = builder.call_tuple(
        ccx, {"ctrl": np.array([f1, f2]), "target": ancilla}
    )
    ancilla, a, x = builder.call_tuple(sub, {"ctrl": ancilla, "x": a, "y": x})
    controls, ancilla = builder.call_tuple(ccx, {"ctrl": controls, "target": ancilla})
    assert isinstance(controls, np.ndarray) and isinstance(ancilla, RegisterBinding)
    f1, f2 = controls
    builder.free(ancilla)
    ancilla = builder.call_dict(initialize, {})["q"]
    controls, ancilla = builder.call_tuple(
        ccx, {"ctrl": np.array([f1, f2]), "target": ancilla}
    )
    ancilla, b, y = builder.call_tuple(plus, {"ctrl": ancilla, "x": b, "y": y})
    controls, ancilla = builder.call_tuple(ccx, {"ctrl": controls, "target": ancilla})
    assert isinstance(controls, np.ndarray) and isinstance(ancilla, RegisterBinding)
    f1, f2 = controls
    builder.free(ancilla)
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    wide = replace(dtype, quantum_bits=2 * dtype.num_qubits)
    ab_word = builder.join(np.concatenate([builder.split(a), builder.split(b)]), wide)
    xy_word = builder.join(np.concatenate([builder.split(x), builder.split(y)]), wide)
    ab, xy, f4 = builder.call_tuple(equals, {"x": ab_word, "y": xy_word, "target": f4})
    assert isinstance(ab, RegisterBinding) and isinstance(xy, RegisterBinding)
    ab_bits = builder.split(ab)
    a_bits, b_bits = np.split(ab_bits, 2)
    a, b = builder.join(a_bits, dtype), builder.join(b_bits, dtype)
    xy_bits = builder.split(xy)
    x_bits, y_bits = np.split(xy_bits, 2)
    x, y = builder.join(x_bits, dtype), builder.join(y_bits, dtype)
    ab_controls = np.concatenate([builder.split(a), builder.split(b)])
    controls, f3 = builder.call_tuple(pair, {"controls": ab_controls, "target": f3})
    assert isinstance(controls, np.ndarray)
    a_bits, b_bits = np.split(controls, 2)
    a, b = builder.join(a_bits, dtype), builder.join(b_bits, dtype)
    xy_controls = np.concatenate([builder.split(x), builder.split(y)])
    controls, junk, out = builder.call_tuple(both, {"ctrl": xy_controls})
    joined_targets = builder.join(np.array([f1, f2]))
    out, targets = builder.call_tuple(copy, {"control": out, "targets": joined_targets})
    assert isinstance(targets, RegisterBinding)
    f1, f2 = builder.split(targets)
    controls = builder.call_dict(
        both_adj, {"ctrl": controls, "junk": junk, "target": out}
    )["ctrl"]
    assert isinstance(controls, np.ndarray)
    x_bits, y_bits = np.split(controls, 2)
    x, y = builder.join(x_bits, dtype), builder.join(y_bits, dtype)
    for flag in (f1, f2, f3, f4, ctrl):
        assert isinstance(flag, RegisterBinding)
        builder.free(flag)
    return builder.finalize({"a": a, "b": b, "x": x, "y": y})
