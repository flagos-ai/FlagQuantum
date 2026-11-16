# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/with_decomposition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/with_decomposition.py

"""Serial and independent fixture shapes in the native register builder."""

from typing import Literal

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture

Mode = Literal["serial", "parallel", "independent"]


def composition_fixture(mode: Mode) -> ProgramRecord:
    atom = AtomicRegisterFixture("x")
    leaf_names = ("atom0", "atom1", "atom2") if mode == "serial" else ("atom",)
    leaves = tuple(
        bit_gate_function(name, tuple(atom.signature), CircuitIR(1, atom.instructions))
        for name in leaf_names
    )
    width = {"serial": 1, "parallel": 3, "independent": 0}[mode]
    builder = ComponentProgramBuilder(Signature.build(reg=width), leaves)
    if mode == "independent":
        for _ in range(3):
            work = builder.allocate(dtype=quantum_bit_type())
            output = builder.call_dict("atom", {"q": work})["q"]
            assert isinstance(output, RegisterBinding)
            builder.free(output)
        return builder.finalize({})

    word = builder.inputs["reg"]
    assert isinstance(word, RegisterBinding)
    if mode == "serial":
        for name in leaf_names:
            output = builder.call_dict(name, {"q": word})["q"]
            assert isinstance(output, RegisterBinding)
            word = output
    else:
        bits = builder.split(word)
        for i, bit in enumerate(bits):
            bits[i] = builder.call_dict("atom", {"q": bit})["q"]
        word = builder.join(bits, word.register.dtype)
    return builder.finalize({"reg": word})
