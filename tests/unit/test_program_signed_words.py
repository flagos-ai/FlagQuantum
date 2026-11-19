"""Execute signed word rebindings with existing linear ownership and codecs."""

from dataclasses import replace

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_rebindings import split_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import signed_register_type
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.test_component_rebindings import NativeRebindings
from tests.unit.test_signed_word_contract import (
    test_signed_word_connections_and_single_consumption as signed_contract,
)

pytestmark = pytest.mark.unit
pytest.importorskip("qualtran")


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 3, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_replaces_source_in_same_signed_contract(
    width: int, reverse: bool
) -> None:
    signed_contract(NativeRebindings, width, reverse)


@pytest.mark.parametrize("width", [1, 3, 8])
def test_signed_split_retains_order_and_full_spectator_state(width: int) -> None:
    record = ProgramRecord((split_word_function("main", signed_register_type(width)),))
    assert verify_program_record(record) == ()
    bound = bind_register_program(record, (tuple(reversed(range(width))),))
    assert bound.outputs == tuple((i,) for i in reversed(range(width)))
    assert bound.circuit.instructions == () and bound.events == ()
    assert lower_register_program(record).functions[0].blocks[0].operations == ()
    state = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(108445),
    )
    state /= state.norm()
    result = (
        Circuit(width + 1, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        replace(signed_register_type(2), quantum_bits=-1),
        replace(signed_register_type(2), quantum_bits=True),
        replace(signed_register_type(2), quantum_bits=1.5),
        replace(signed_register_type(2), classical_bits=1),
        replace(signed_register_type(2), parameters=(("unresolved", 1),)),
    ],
)
def test_signed_invalid_execution_profiles_remain_refused(dtype: RegisterType) -> None:
    with pytest.raises(ValueError, match="executable"):
        executable_register_width(dtype)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 3, 4, 8])
def test_signed_source_codec_and_split_join(width: int) -> None:
    from qualtran import QInt
    from qualtran.bloqs.bookkeeping import Join, Split

    from flagquantum.simulation.integer_encoding import (
        integer_from_bits,
        integer_to_bits,
    )

    source = QInt(width)
    for value in range(-(1 << (width - 1)), 1 << (width - 1)):
        bits = Split(source).on_classical_vals(reg=value)["reg"]
        assert integer_to_bits(value, width, signed=True) == list(bits)
        assert (
            integer_from_bits(bits, width, signed=True)
            == Join(source).on_classical_vals(reg=bits)["reg"]
            == value
        )


def test_zero_width_signed_word_retains_logical_ownership() -> None:
    record = ProgramRecord((split_word_function("main", signed_register_type(0)),))
    assert executable_register_width(signed_register_type(0)) == 0
    assert verify_program_record(record) == ()
    assert lower_register_program(record).functions[0].blocks[0].operations == ()
    with pytest.raises(ValueError, match="at least one physical wire"):
        bind_register_program(record, ((),))
    bound = bind_register_program(record, ((),), clean=(0,))
    assert bound.outputs == ()
    assert bound.circuit.instructions == () and bound.events == ()


@pytest.mark.parametrize("width", [1, 3, 8])
def test_signed_workspace_lifetime_uses_existing_clean_pool(width: int) -> None:
    from qualtran import BloqBuilder, QInt

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.signatures import Signature

    original = BloqBuilder()
    source_word = original.allocate(dtype=QInt(width))
    original.free(source_word)
    source = original.finalize()
    builder = ComponentProgramBuilder(Signature(()))
    word = builder.allocate(dtype=signed_register_type(width))
    assert word.register.dtype == signed_register_type(width)
    builder.free(word)
    record = builder.finalize({})
    assert len(source.connections) == 1
    assert len(record.functions[0].blocks[0].operations) == 2
    bound = bind_register_program(record, (), clean=tuple(range(width)))
    assert bound.outputs == () and bound.circuit.instructions == ()
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", tuple(range(width)), False),
        ("free", tuple(range(width)), False),
    ]
