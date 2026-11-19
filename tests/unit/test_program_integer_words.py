"""Bind canonical integer storage without changing its bit representation."""

from dataclasses import replace
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_rebindings import split_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    montgomery_register_type,
    ones_complement_register_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_register_width,
    logical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_rebindings import NativeRebindings
from tests.unit.test_integer_word_contract import (
    INTEGER_WORDS,
)
from tests.unit.test_integer_word_contract import (
    test_integer_storage_connections_and_consumption as storage_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dtype", INTEGER_WORDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_native_replaces_original_integer_contract(
    dtype: RegisterType, reverse: bool
) -> None:
    storage_contract(NativeRebindings, dtype, reverse)


@pytest.mark.parametrize("dtype", INTEGER_WORDS)
def test_physical_split_retains_storage_and_complex_spectator(
    dtype: RegisterType,
) -> None:
    width = int(dtype.num_qubits)
    record = ProgramRecord((split_word_function("main", dtype),))
    assert verify_program_record(record) == ()
    bound = bind_register_program(record, (tuple(reversed(range(width))),))
    assert bound.outputs == tuple((i,) for i in reversed(range(width)))
    assert bound.events == () and bound.circuit.instructions == ()
    assert lower_register_program(record).functions[0].blocks[0].operations == ()
    state = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(1084550),
    )
    state /= state.norm()
    actual = (
        Circuit(width + 1, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", INTEGER_WORDS)
def test_allocate_free_preserves_integer_descriptor(dtype: RegisterType) -> None:
    from qualtran import BloqBuilder

    from tests.unit.test_register_bindings import SourceContract

    original = BloqBuilder()
    source_register: Any = SourceContract.register
    original.free(
        original.allocate(dtype=source_register(Register("word", dtype)).dtype)
    )
    source = original.finalize()
    builder = ComponentProgramBuilder(Signature(()))
    word = builder.allocate(dtype=dtype)
    assert word.register.dtype == dtype
    builder.free(word)
    record = builder.finalize({})
    assert verify_program_record(record) == ()
    assert len(source.connections) == 1
    width = int(dtype.num_qubits)
    bound = bind_register_program(record, (), clean=tuple(range(width)))
    assert bound.outputs == () and bound.circuit.instructions == ()
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", tuple(range(width)), False),
        ("free", tuple(range(width)), False),
    ]


@pytest.mark.parametrize(
    "dtype",
    [
        ones_complement_register_type(0),
        bounded_unsigned_register_type(0, 1),
        montgomery_register_type(0),
    ],
)
def test_zero_width_integer_storage_retains_owned_word(dtype: RegisterType) -> None:
    record = ProgramRecord((split_word_function("main", dtype),))
    assert verify_program_record(record) == ()
    assert executable_register_width(dtype) == 0
    assert lower_register_program(record).functions[0].blocks[0].operations == ()
    bound = bind_register_program(record, ((),), clean=(0,))
    assert bound.outputs == () and bound.circuit.instructions == ()


@pytest.mark.parametrize("dtype", INTEGER_WORDS)
@pytest.mark.parametrize(
    "field,value",
    [
        ("quantum_bits", -1),
        ("quantum_bits", True),
        ("quantum_bits", 2.5),
        ("classical_bits", 1),
        ("parameters", (("unknown", 1),)),
    ],
)
def test_malformed_integer_profiles_refused(
    dtype: RegisterType, field: str, value: Any
) -> None:
    malformed = replace(dtype, **{field: value})
    with pytest.raises(ValueError):
        executable_register_width(malformed)


@pytest.mark.parametrize(
    "dtype",
    [
        bounded_unsigned_register_type(3, sympy.Symbol("b", integer=True)),
        montgomery_register_type(3, sympy.Symbol("p", integer=True)),
        ones_complement_register_type(sympy.Symbol("n", integer=True)),
    ],
)
def test_symbolic_integer_declarations_do_not_claim_physical_binding(
    dtype: RegisterType,
) -> None:
    assert logical_register_width(dtype) == dtype.num_qubits
    with pytest.raises(ValueError):
        executable_register_width(dtype)
