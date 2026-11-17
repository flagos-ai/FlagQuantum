"""Check the locked Cast signature and owned classical-bit aliasing."""

from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest

from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.layout.register_cast import cast_signature
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    fixed_point_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.bookkeeping import cast

    raw = Path(cast.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "9fd7bee565a88e97a4aa1b1c7945c5c6529d4381"
    )


@pytest.mark.parametrize(
    "fixture", ["shaped", "symbolic", "classical", "readout", "load"]
)
def test_cast_metadata_preserves_type_shape_and_resource_width(fixture):
    from qualtran import CBit, QBit, QFxp, QInt, QUInt
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    source, target, src_dtype, dst_dtype, shape = {
        "shaped": (
            QInt(4),
            QFxp(4, 2, signed=True),
            signed_register_type(4),
            fixed_point_register_type(4, 2, signed=True),
            (2, 2),
        ),
        "symbolic": (
            QInt(n),
            QUInt(5),
            signed_register_type(n),
            unsigned_register_type(5),
            (2, 3),
        ),
        "classical": (CBit(), CBit(), classical_bit_type(), classical_bit_type(), 2),
        "readout": (QBit(), CBit(), quantum_bit_type(), classical_bit_type(), (2,)),
        "load": (CBit(), QBit(), classical_bit_type(), quantum_bit_type(), (2,)),
    }[fixture]
    opt_in = fixture in {"readout", "load"}
    locked = Cast(source, target, shape=shape, allow_quantum_to_classical=opt_in)
    signature = cast_signature(
        src_dtype, dst_dtype, shape, allow_quantum_to_classical=opt_in
    )
    assert tuple(reg.dtype for reg in signature) == (src_dtype, dst_dtype)
    assert tuple(
        (
            reg.name,
            reg.dtype.num_qubits,
            reg.dtype.num_cbits,
            reg.shape_symbolic,
            reg.side.name,
        )
        for reg in signature
    ) == tuple(
        (
            reg.name,
            reg.dtype.num_qubits,
            reg.dtype.num_cbits,
            reg.shape_symbolic,
            reg.side.name,
        )
        for reg in locked.signature
    )
    assert signature.n_qubits() == get_cost_value(locked, QubitCount())
    assert get_cost_value(locked, QECGatesCost()).asdict() == {}


def test_cast_signature_keeps_width_check_order_and_zero_classical_identity():
    from qualtran import QAny, QInt, QUInt
    from qualtran._infra.data_types import CUInt
    from qualtran.bloqs.bookkeeping import Cast
    from sympy import Symbol

    with pytest.raises(ValueError, match="preserve the number of bits"):
        Cast(QInt(5), QUInt(Symbol("n")))
    with pytest.raises(ValueError, match="preserve the word width"):
        cast_signature(signed_register_type(5), unsigned_register_type(Symbol("n")))
    with pytest.raises(ValueError, match="potentially dangerous"):
        Cast(QAny(0), CUInt(0))
    with pytest.raises(ValueError, match="explicit opt-in"):
        cast_signature(unsigned_register_type(0), classical_unsigned_register_type(0))


def test_owned_cbit_cast_preserves_existing_slots_without_cast_instructions():
    from qualtran import CBit
    from qualtran.bloqs.bookkeeping import Cast

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.signatures import Signature

    dtype = classical_bit_type()
    leaf = cast_word_function("cast_bit", dtype, dtype, (2,))
    block = leaf.blocks[0]
    assert block.operations == () and block.arguments == block.return_values
    assert leaf.registers is not None
    builder = ComponentProgramBuilder(Signature(leaf.registers), (leaf,))
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    identity = ComponentProgramBuilder(Signature(leaf.registers))
    baseline_program = identity.finalize(identity.inputs)
    for inputs in product((0, 1), repeat=2):
        expected = tuple(
            Cast(CBit(), CBit()).on_classical_vals(reg=i)["reg"] for i in inputs
        )
        baseline = bind_register_program(
            baseline_program,
            ((), ()),
            borrowed=(0,),
            classical_inputs=expected,
        )
        actual = bind_register_program(
            record, ((), ()), borrowed=(0,), classical_inputs=inputs
        )
        assert actual.outputs == ((), ())
        assert actual.classical_outputs == baseline.classical_outputs == ((0,), (1,))
        assert actual.circuit.instructions == baseline.circuit.instructions
        assert actual.events == baseline.events == ()
