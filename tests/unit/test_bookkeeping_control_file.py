"""Compare complete inherited bookkeeping control with original descriptors."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.control.component_bookkeeping import (
    bookkeeping_component_control_system,
    call_with_unchanged_controls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_fixed_point_register_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
    fixed_point_register_type,
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.test_component_graph import _native_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


def original_case(kind: str) -> tuple[Any, dict[Any, RegisterType]]:
    from qualtran import QGF, CInt, CUInt, QAny, QBit, QFxp, QGFPoly, QInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Partition, Split
    from qualtran.bloqs.bookkeeping.partition import Join2, Split2
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        GFPolyJoin,
        GFPolySplit,
    )

    n = sympy.Symbol("n")
    types = {
        QBit(): quantum_bit_type(),
        QAny(n): opaque_register_type(n),
        QAny(2): opaque_register_type(2),
        QAny(n + 2): opaque_register_type(n + 2),
        QInt(n): signed_register_type(n),
        QFxp(n, 2): fixed_point_register_type(n, 2),
        QFxp(3, 1, signed=True): fixed_point_register_type(3, 1, True),
        QFxp(0, -2): fixed_point_register_type(0, -2),
        CUInt(3): classical_unsigned_register_type(3),
        CInt(3): classical_signed_register_type(3),
    }
    if kind in {"polynomial_split", "polynomial_join"}:
        field = FiniteFieldEncoding(2, 2)
        polynomial = FiniteFieldPolynomialEncoding(2, field)
        dtype = QGFPoly(2, QGF(2, 2))
        types[dtype] = polynomial.register_type()
        types[dtype.qgf] = field.register_type()
        return (
            GFPolyJoin(dtype) if kind.endswith("join") else GFPolySplit(dtype)
        ), types
    cases = {
        "allocate": Allocate(QFxp(n, 2)),
        "free": Free(QInt(n)),
        "split": Split(QFxp(3, 1, signed=True)),
        "join": Join(QFxp(0, -2)),
        "cast": Cast(CUInt(3), CInt(3), shape=(2,)),
        "partition": Partition(
            n + 2, (SourceRegister("a", QAny(n)), SourceRegister("b", QAny(2)))
        ),
        "split2": Split2(n, 2),
        "join2": Join2(n, 2),
    }
    return cases[kind], types


def declaration(source: Any, types: dict[Any, RegisterType]) -> ProgramRecord:
    registers = tuple(
        Register(
            reg.name, types[reg.dtype], reg.shape_symbolic, RegisterSide[reg.side.name]
        )
        for reg in source.signature
    )
    return ProgramRecord((FunctionRecord("leaf", (), registers=registers),))


@pytest.mark.parametrize(
    "kind",
    (
        "allocate",
        "free",
        "split",
        "join",
        "cast",
        "partition",
        "split2",
        "join2",
        "polynomial_split",
        "polynomial_join",
    ),
)
@pytest.mark.parametrize("container", ("tuple", "list", "empty"))
def test_actual_ten_descendants_generic_logical_control_and_order(
    kind: str, container: str
) -> None:
    from qualtran import BloqBuilder, CtrlSpec, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    source, types = original_case(kind)
    record = declaration(source, types)
    selected, adder = bookkeeping_component_control_system(record, "leaf")
    assert selected is record and verify_program_record(record) == ()
    registers = record.functions[0].registers
    assert registers is not None
    controls_port = Register("controls", quantum_bit_type(), (2,))
    builder = ComponentProgramBuilder(
        Signature((controls_port, *registers)), record.functions, name="caller"
    )
    original_builder, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (SourceRegister("controls", QBit(), shape=(2,)), *source.signature)
        )
    )
    values = {
        reg.name: builder.inputs[reg.name] for reg in Signature(registers).lefts()
    }
    source_values = {reg.name: inputs[reg.name] for reg in source.signature.lefts()}
    before = dict(values)
    native_controls: Sequence[BindingCollection]
    returned: Sequence[BindingCollection]
    if container == "list":
        native_controls = [builder.inputs["controls"]]
        original_controls: Any = [inputs["controls"]]
        returned, outputs = call_with_unchanged_controls(
            builder, "leaf", native_controls, values
        )
    else:
        native_controls = () if container == "empty" else (builder.inputs["controls"],)
        original_controls = () if container == "empty" else (inputs["controls"],)
        returned, outputs = adder(builder, native_controls, values)
    unchanged, callback = source.get_ctrl_system(CtrlSpec(cvs=(0, 1)))
    original_returned, original_outputs = callback(
        original_builder, original_controls, source_values
    )
    assert (
        unchanged is source
        and returned is native_controls
        and original_returned is original_controls
    )
    assert all(values[key] is value for key, value in before.items())
    right_names = tuple(reg.name for reg in Signature(registers).rights())
    module = builder.finalize(
        {
            "controls": builder.inputs["controls"],
            **dict(zip(right_names, outputs, strict=True)),
        }
    )
    composite = original_builder.finalize(
        controls=inputs["controls"],
        **dict(zip(right_names, original_outputs, strict=True)),
    )
    assert verify_program_record(module) == ()
    operation = module.functions[0].blocks[0].operations[0]
    assert (
        operation.callee == "leaf"
        and len(module.functions[0].blocks[0].operations) == 1
    )
    instances = tuple(composite.iter_bloqsoqs())
    assert len(instances) == 1 and instances[0][0].bloq is source
    trace_row = _native_trace(module, "caller")[0]
    assert isinstance(trace_row, tuple)
    actual_inputs = trace_row[1]
    expected_inputs = tuple(
        (
            name,
            (
                np.asarray(value).shape,
                tuple(
                    ("entry", soq.reg.name, soq.idx) for soq in np.asarray(value).flat
                ),
            ),
        )
        for name, value in instances[0][1].items()
    )
    assert actual_inputs == expected_inputs
    graph = program_component_call_graph(module, "caller", max_depth=1)
    assert graph.direct_calls == {("caller", "leaf"): 1}
    assert graph.leaf_counts == {"leaf": 1}
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(module, (), entry="caller")


@pytest.mark.parametrize(
    "dtype_kind",
    (
        "unsigned",
        "signed",
        "classical_unsigned",
        "classical_signed",
        "fixed",
        "signed_fixed",
        "empty_fixed",
        "classical_fixed",
        "cbit",
    ),
)
def test_same_control_protocol_ignores_dtype_and_missing_control_spec(
    dtype_kind: str,
) -> None:
    from qualtran import BloqBuilder, CBit, CFxp, CInt, CUInt, QFxp, QInt, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.bookkeeping import Cast

    pairs = {
        "unsigned": (QUInt(3), RegisterType("uint", 3)),
        "signed": (QInt(3), signed_register_type(3)),
        "classical_unsigned": (CUInt(3), classical_unsigned_register_type(3)),
        "classical_signed": (CInt(3), classical_signed_register_type(3)),
        "fixed": (QFxp(3, -2), fixed_point_register_type(3, -2)),
        "signed_fixed": (
            QFxp(3, 1, signed=True),
            fixed_point_register_type(3, 1, True),
        ),
        "empty_fixed": (QFxp(0, -2), fixed_point_register_type(0, -2)),
        "classical_fixed": (CFxp(3, 1), classical_fixed_point_register_type(3, 1)),
        "cbit": (CBit(), classical_bit_type()),
    }
    original_type, dtype = pairs[dtype_kind]
    source = Cast(original_type, original_type, shape=(2,))
    record = declaration(source, {original_type: dtype})
    _, adder = bookkeeping_component_control_system(record, "leaf")
    registers = record.functions[0].registers
    assert registers is not None
    builder = ComponentProgramBuilder(
        Signature(registers), record.functions, name="caller"
    )
    original_builder, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("reg", original_type, shape=(2,)),))
    )
    controls: tuple[BindingCollection, ...] = ()
    returned, outputs = adder(builder, controls, builder.inputs)
    selected, original_adder = source.get_ctrl_system()
    source_controls: tuple[Any, ...] = ()
    original_returned, original_outputs = original_adder(
        original_builder, source_controls, inputs
    )
    assert (
        selected is source
        and returned is controls
        and original_returned is source_controls
    )
    module = builder.finalize({"reg": outputs[0]})
    original = original_builder.finalize(reg=original_outputs[0])
    assert verify_program_record(module) == ()
    assert len(original.bloq_instances) == 1
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(module, (), entry="caller")


def test_declaration_cannot_hide_a_computational_body_after_replacement() -> None:
    from tests.unit.test_register_on_each import bit_child

    registers = bit_child("x").registers
    record = ProgramRecord((FunctionRecord("main", (), registers=registers),))
    assert bookkeeping_component_control_system(record)[0] is record
    implemented = replace(record, functions=(replace(bit_child("x"), name="main"),))
    with pytest.raises(ValueError, match="computational"):
        bookkeeping_component_control_system(implemented)
