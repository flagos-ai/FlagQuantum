"""Typed word storage, owned reads and nested transport use existing slots."""

from dataclasses import replace
from typing import Any

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from flagquantum.compiler.registers.classical_word_storage import classical_word_bits
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.classical_dataflow import parse_classical_xor
from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
)
from flagquantum.core.ir.program.register_types import (
    classical_bounded_unsigned_register_type,
    classical_fixed_point_register_type,
    classical_montgomery_register_type,
    classical_ones_complement_register_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_register_codec import encode_register_value
from tests.unit.test_program_classical_registers import identity

pytestmark = pytest.mark.unit
CASES = (
    (classical_unsigned_register_type(3), tuple(range(8))),
    (classical_signed_register_type(3), tuple(range(-4, 4))),
    (classical_ones_complement_register_type(3), tuple(range(-3, 4))),
    (classical_bounded_unsigned_register_type(3, 5), tuple(range(5))),
    (classical_montgomery_register_type(3, 5), tuple(range(8))),
    (classical_montgomery_register_type(3), tuple(range(8))),
    (classical_fixed_point_register_type(3, 1), tuple(range(8))),
    (classical_fixed_point_register_type(3, 1, True), tuple(range(-4, 4))),
)


@pytest.mark.parametrize("dtype,values", CASES)
def test_shared_physical_encoder_matches_numerical_owner(
    dtype: RegisterType, values: tuple[int, ...]
) -> None:
    for value in values:
        expected = tuple(encode_register_value(Register("c", dtype), value))
        assert classical_word_bits(dtype, value) == expected


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype,values", CASES)
def test_storage_bits_match_unmodified_source(
    dtype: RegisterType, values: tuple[int, ...]
) -> None:
    from qualtran import CFxp, CInt, CIntOnesComp, CMontgomeryUInt, CUInt
    from qualtran._infra.data_types import _BUInt

    parameters = dict(dtype.parameters)
    source: Any
    if "bound" in parameters:
        # Qualtran has a bounded integer codec, without a CBUInt descriptor.
        source = _BUInt(3, 5)
    elif dtype.encoding == "fixed_point":
        source = CFxp(3, 1, signed=parameters["signed"])
    elif dtype.encoding == "montgomery":
        source = CMontgomeryUInt(3, parameters.get("modulus"))
    else:
        source = {"uint": CUInt, "signed": CInt, "ones_complement": CIntOnesComp}[
            dtype.encoding
        ](3)
    for value in values:
        assert classical_word_bits(dtype, value) == tuple(source.to_bits(value))


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_mixed_typed_words_forward_through_repeated_stored_calls(
    shape: tuple[int, ...],
) -> None:
    ports = (
        Register("signed", classical_signed_register_type(3), shape),
        Register("montgomery", classical_montgomery_register_type(3, 5)),
        Register("empty", classical_unsigned_register_type(0)),
        Register("q", quantum_bit_type()),
    )
    leaf = identity("leaf", ports)
    builder = ComponentProgramBuilder(Signature(ports), (leaf,))
    carried = builder.inputs
    for _ in range(3):
        carried = builder.call_dict("leaf", carried)
    record = builder.finalize(carried)
    count = len(tuple(ports[0].all_idxs()))
    values = (*(-2 for _ in range(count)), 7, 0)
    bound = bind_register_program(
        record, ((),) * (count + 2) + ((0,),), classical_inputs=(*values, None)
    )
    bits = tuple(bit for value in values[:-1] for bit in (1, 1, int(value == 7)))
    assert (
        tuple(parse_classical_xor(g).constant for g in bound.circuit.instructions)
        == bits
    )
    assert bound.outputs == ((),) * (count + 2) + ((0,),)
    assert bound.classical_outputs == (
        *(tuple(range(i, i + 3)) for i in range(0, 3 * (count + 1), 3)),
        (),
        (),
    )
    assert bound.events == ()


@pytest.mark.parametrize("width", [0, 1, 3])
def test_repeated_reads_share_msb_snapshots_and_emit_no_slots(width: int) -> None:
    c, q, first, second, output = (ValueRef("entry", i, "register") for i in range(5))
    q0, q1, q2 = (ValueRef("entry", i, "qubit") for i in (10, 11, 12))
    a = tuple(ValueRef("entry", 20 + i, "bool") for i in range(width))
    b = tuple(ValueRef("entry", 20 + width + i, "bool") for i in range(width))
    operations = [
        Operation("read_classical_register", (c,), (first, *a)),
        Operation("read_classical_register", (first,), (second, *b)),
    ]
    if width:
        operations.extend(
            (
                Operation("split_register", (q,), (q0,)),
                Operation(
                    "x", (q0,), (q1,), condition_clauses=((ConditionRef(a[0], True),),)
                ),
                Operation(
                    "h", (q1,), (q2,), condition_clauses=((ConditionRef(b[0], True),),)
                ),
                Operation(
                    "join_register", (q2,), (output,), register_type=quantum_bit_type()
                ),
            )
        )
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (c, q),
                operations=tuple(operations),
                return_values=(second, output if width else q),
            ),
        ),
        registers=(
            Register("c", classical_unsigned_register_type(width)),
            Register("q", quantum_bit_type()),
        ),
    )
    bound = bind_register_program(
        ProgramRecord((function,)),
        ((), (0,)),
        classical_inputs=((1 << width) - 1, None),
    )
    assert bound.classical_outputs == (tuple(range(width)), ())
    assert len(bound.circuit.instructions) == width + (2 if width else 0)
    assert all(
        g.metadata["conditions"] == ((0, 1),)
        for g in bound.circuit.instructions[width:]
    )
    assert bound.events == ()
    # Owned handles cannot be used again after either read.
    block = function.blocks[0]
    stale = replace(
        function, blocks=(replace(block, return_values=(c, output if width else q)),)
    )
    with pytest.raises(ValueError, match="linear_duplicate"):
        bind_register_program(
            ProgramRecord((stale,)), ((), (0,)), classical_inputs=(0, None)
        )


@pytest.mark.parametrize("dtype,values", CASES)
def test_outside_domains_and_nonbuiltin_inputs_fail_before_binding(
    dtype: RegisterType, values: tuple[int, ...]
) -> None:
    function = identity(
        "main", (Register("c", dtype), Register("q", quantum_bit_type()))
    )
    invalid: Any
    for invalid in (min(values) - 1, max(values) + 1, True, 1.0, np.int64(1), None):
        with pytest.raises(ValueError):
            bind_register_program(
                ProgramRecord((function,)), ((), (0,)), classical_inputs=(invalid, None)
            )


@pytest.mark.ftqc_reference
def test_zero_width_identity_and_empty_source_defects_are_explicit() -> None:
    from qualtran import BloqBuilder, CUInt, RightDangle
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    source = CUInt(0)
    # The unchanged scalar codec formats one zero despite the declared zero width.
    assert source.to_bits(0) == [0]
    assert classical_word_bits(classical_unsigned_register_type(0), 0) == ()
    for shape in ((), (0,)):
        builder, inputs = BloqBuilder.from_signature(
            SourceSignature((SourceRegister("c", source, shape),))
        )
        graph = builder.finalize(**inputs)
        if shape:
            with pytest.raises(KeyError) as failure:
                graph.call_classically(c=np.empty(0, dtype=int))
            assert failure.value.args == (RightDangle,)
        else:
            assert graph.call_classically(c=0) == (0,)
