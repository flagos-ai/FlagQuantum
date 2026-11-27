# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/many_registers.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/many_registers.py

"""Shaped and typed register fixtures using the native Program builder."""

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture


def multi_register_fixture() -> ProgramRecord:
    """Apply X, four row-major CX calls and a spectator split/rejoin."""
    bit, pair, triple = (
        quantum_bit_type(),
        opaque_register_type(2),
        opaque_register_type(3),
    )
    x, cx = AtomicRegisterFixture("x"), AtomicRegisterFixture("cx")
    signature = Signature(
        (Register("xx", bit), Register("yy", pair, (2, 2)), Register("zz", triple))
    )
    builder = ComponentProgramBuilder(
        signature,
        (
            bit_gate_function("atom", tuple(x.signature), CircuitIR(1, x.instructions)),
            bit_gate_function(
                "two_bit", tuple(cx.signature), CircuitIR(2, cx.instructions)
            ),
        ),
    )
    xx = builder.call_dict("atom", {"q": builder.inputs["xx"]})["q"]
    pairs, zz = builder.inputs["yy"], builder.inputs["zz"]
    assert isinstance(pairs, np.ndarray) and isinstance(zz, RegisterBinding)
    outputs = np.empty((2, 2), dtype=object)
    for index in np.ndindex(pairs.shape):
        pair_word = pairs[index]
        assert isinstance(pair_word, RegisterBinding)
        ctrl, target = builder.split(pair_word)
        changed = builder.call_dict("two_bit", {"ctrl": ctrl, "target": target})
        outputs[index] = builder.join((changed["ctrl"], changed["target"]), pair)
    return builder.finalize(
        {"xx": xx, "yy": outputs, "zz": builder.join(builder.split(zz), triple)}
    )


def companion_signatures() -> tuple[Signature, Signature, Signature]:
    bounded = bounded_unsigned_register_type(4, 3)
    signed_fraction = fixed_point_register_type(8, 6, signed=True)
    unsigned_fraction = fixed_point_register_type(8, 8)
    return (
        Signature.build_from_dtypes(xx=bounded, yy=signed_fraction),
        Signature.build_from_dtypes(xx=signed_fraction, yy=unsigned_fraction),
        Signature.build_from_dtypes(
            a=bounded,
            b=signed_fraction,
            c=unsigned_fraction,
            d=unsigned_register_type(8),
        ),
    )
