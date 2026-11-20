# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 for_testing/casting.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/casting.py

"""Check the original Cast/Add/Cast fixture through native typed calls."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.signatures import Signature


def cast_add_fixture(width: int) -> ProgramRecord:
    unsigned = unsigned_register_type(width)
    fraction = fixed_point_register_type(width, width)
    add = addition_component_program(width, name="add")
    to_unsigned = cast_word_function("to_unsigned", fraction, unsigned)
    to_fraction = cast_word_function("to_fraction", unsigned, fraction)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=unsigned, b=fraction),
        (to_unsigned, *add.functions, to_fraction),
    )
    converted = builder.call_dict("to_unsigned", {"reg": builder.inputs["b"]})["reg"]
    added = builder.call_dict("add", {"a": builder.inputs["a"], "b": converted})
    restored = builder.call_dict("to_fraction", {"reg": added["b"]})["reg"]
    return builder.finalize({"a": added["a"], "b": restored})


@pytest.mark.unit
@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 4])
def test_typed_cast_add_calls_preserve_fixed_point_storage(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing import TestCastToFrom

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source
    from tests.unit.test_subtraction_foreign_file import native_signature

    source = TestCastToFrom(bitsize=width)
    record = cast_add_fixture(width)
    main = next(fn for fn in record.functions if fn.name == "main")
    assert main.registers == tuple(native_signature(source.signature))
    assert main.registers is not None
    assert [(r.name, r.bitsize, r.side.name) for r in main.registers] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    labels = {"to_unsigned": "Cast", "add": "Add", "to_fraction": "Cast"}
    assert [
        labels[op.callee] for op in main.blocks[0].operations if op.callee is not None
    ] == [
        type(node.bloq).__name__
        for node, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ]
    a, b = tuple(range(width)), tuple(range(width, 2 * width))
    work = tuple(range(2 * width, 3 * width - 1))
    bound = bind_register_program(record, (a, b), clean=work)
    assert bound.outputs == (a, b)
    instructions = bound.circuit.instructions
    slots = {
        op.wires[0]: op.metadata["classical_bit"]
        for op in instructions
        if op.name == "measure"
    }
    oracle = reference_arithmetic_gates(source, {"a": a, "b": b}, work, slots)
    assert_source(instructions, oracle)

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from tests.unit.measurement_checks import forced_branch

    values = torch.arange(1, (1 << (2 * width)) + 1, dtype=torch.float64)
    ket = torch.complex(values, values.remainder(7) - 3)
    ket /= ket.norm()
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    initial = torch.kron(ket, clean)
    expected = torch.kron(torch.as_tensor(source.tensor_contract()) @ ket, clean)
    for history in product((0, 1), repeat=len(slots)):
        actual = forced_branch(initial, instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (len(slots) / 2), atol=3e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << len(work))[:, 1:]) == 0
    if not slots:
        circuit = Circuit(2 * width + len(work), inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(2 * width + len(work), instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=3e-12, rtol=0
        )
