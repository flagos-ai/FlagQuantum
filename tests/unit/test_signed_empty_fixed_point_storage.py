"""Proposal 116 stores raw signed/empty fixed words without arithmetic changes."""

from math import prod

import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,signed", ((0, False), (2, True), (3, True), (8, True)))
@pytest.mark.parametrize("fraction", (-7, -2, 0))
@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (0, 2)))
def test_cast_split_join_retains_metadata_shape_and_raw_word_ownership(
    width: int, signed: bool, fraction: int, shape: tuple[int, ...]
) -> None:
    from qualtran import QFxp, QUInt
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype = fixed_point_register_type(width, fraction, signed)
    original = dtype
    assert executable_register_width(dtype) == width
    unsigned = RegisterType("uint", width)
    signature = Signature((Register("reg", dtype, shape),))
    builder = ComponentProgramBuilder(
        signature,
        (
            cast_word_function("unsigned_view", dtype, unsigned, shape),
            cast_word_function("fixed_view", unsigned, dtype, shape),
        ),
    )
    view = builder.call_dict("unsigned_view", builder.inputs)
    output = builder.call_dict("fixed_view", view)
    program = builder.finalize(output)
    assert not verify_program_record(program)
    assert program.functions[0].registers == tuple(signature)
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(prod(shape)))
    bound = bind_register_program(program, words, borrowed=(prod(shape) * width,))
    assert bound.outputs == words
    assert bound.circuit.instructions == () and bound.events == ()
    assert dtype == original
    source_dtype = QFxp(width, fraction, signed=signed)
    source = Cast(source_dtype, QUInt(width), shape)
    cost = get_cost_value(source, QECGatesCost())
    assert cost.total_t_count() == cost.clifford == cost.measurement == 0
    assert source.signature[0].bitsize == width


@pytest.mark.parametrize(
    "width,fraction,signed",
    ((0, 0, False), (0, -2, False), (3, 1, True), (3, -2, True)),
)
def test_create_and_consume_preserve_caller_spectator_and_empty_handles(
    width: int, fraction: int, signed: bool
) -> None:
    dtype = fixed_point_register_type(width, fraction, signed)
    create = ComponentProgramBuilder(
        Signature((Register("result", dtype, side=RegisterSide.RIGHT),))
    )
    result = create.allocate(dtype=dtype)
    forward = bind_register_program(
        create.finalize({"result": result}),
        (),
        clean=tuple(range(width)),
        borrowed=(width,),
        allow_workspace_outputs=True,
    )
    assert forward.outputs == (tuple(range(width)),)
    consume = ComponentProgramBuilder(
        Signature((Register("result", dtype, side=RegisterSide.LEFT),))
    )
    word = consume.inputs["result"]
    assert isinstance(word, RegisterBinding)
    consume.free(consume.join(consume.split(word), dtype))
    inverse = bind_register_program(
        consume.finalize({}),
        forward.outputs,
        borrowed=(width,),
        input_ownership=tuple((wire, False) for wire in range(width)),
    )
    assert inverse.outputs == ()
    assert forward.circuit.instructions == inverse.circuit.instructions == ()
    assert all(width not in event.wires for event in (*forward.events, *inverse.events))


@pytest.mark.parametrize("fraction", (0, -2))
@pytest.mark.parametrize("inverse", (False, True))
def test_empty_source_word_complex_operator_in_a_real_spectator_context(
    fraction: int, inverse: bool
) -> None:
    from qualtran import BloqBuilder, QBit, QFxp
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, OnEach, TGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.simulation.unitary import get_unitary
    from tests.unit.test_on_each_foreign_file import native_declaration
    from tests.unit.test_register_on_each import bit_child

    dtype = QFxp(0, fraction)
    original = OnEach(0, TGate(), dtype)
    source = original.adjoint() if inverse else original
    assert len(source.decompose_bloq().bloq_instances) == 2
    # Generic source Adjoint has no OnEach constructor fields; its empty THRU
    # signature is unchanged and contains no child invocation to reverse.
    assert source.signature == original.signature
    program = native_declaration(
        original, bit_child("tdg" if inverse else "t")
    ).program()
    bound = bind_register_program(program, ((),), borrowed=(0,))
    assert (
        bound.outputs == ((),) and bound.circuit.instructions == () and not bound.events
    )
    builder, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (SourceRegister("q", dtype), SourceRegister("spectator", QBit()))
        )
    )
    q = builder.add(source, q=inputs["q"])
    spectator = builder.add(Hadamard(), q=inputs["spectator"])
    spectator = builder.add(TGate(), q=spectator)
    reference = builder.finalize(q=q, spectator=spectator)
    circuit = Circuit(1, dtype=torch.complex128).h(0).t(0)
    circuit.append(FieldOperation(1, bound.circuit.instructions))
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(reference.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 0


@pytest.mark.parametrize("width,fraction", ((2, 1), (3, -2)))
@pytest.mark.parametrize("activation", (0, 1))
def test_signed_control_keeps_source_and_native_refusal(
    width: int, fraction: int, activation: int
) -> None:
    from qualtran import CtrlSpec, QFxp
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    from examples.algorithms.controlled_programs import controlled_x_program
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    dtype = fixed_point_register_type(width, fraction, True)
    assert executable_register_width(dtype) == width
    with pytest.raises(NotImplementedError, match="only supports integer types"):
        CtrlSpecAnd(CtrlSpec(QFxp(width, fraction, signed=True), activation))
    with pytest.raises(ValueError, match="single-bit"):
        controlled_x_program(
            (Register("ctrl", dtype),), (ControlRegister(width, (activation,)),)
        )
