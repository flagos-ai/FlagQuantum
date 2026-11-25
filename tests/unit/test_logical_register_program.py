"""Source multi-typed calls and symbolic graphs through native construction."""

from dataclasses import replace

import numpy as np
import pytest

from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_graph import (
    copy_component_program,
    iter_component_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
sympy = pytest.importorskip("sympy")


def _typed_builder() -> ComponentProgramBuilder:
    types = (
        bounded_unsigned_register_type(4, 3),
        fixed_point_register_type(8, 6, True),
        fixed_point_register_type(8, 8),
        RegisterType("uint", 8),
    )
    signature = Signature(
        tuple(Register(name, dtype) for name, dtype in zip("abcd", types, strict=True))
    )
    multi = FunctionRecord("multi", (), registers=tuple(signature))
    fixed = FunctionRecord(
        "fixed", (), registers=(Register("xx", types[1]), Register("yy", types[2]))
    )
    return ComponentProgramBuilder(signature, (multi, fixed))


@pytest.mark.parametrize("right", ["c", "d"])
def test_original_multityped_success_preserves_producer_types(right: str) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.for_testing.many_registers import (
        TestMultiTypedRegister,
        TestQFxp,
    )

    source, source_inputs = BloqBuilder.from_signature(
        TestMultiTypedRegister().signature, add_registers_allowed=True
    )
    source_outputs = source.add_d(TestMultiTypedRegister(), **source_inputs)
    source_result = source.add_d(
        TestQFxp(), xx=source_outputs["b"], yy=source_outputs[right]
    )
    source_outputs["b"], source_outputs[right] = (
        source_result["xx"],
        source_result["yy"],
    )
    reference = source.finalize(**source_outputs)
    builder = _typed_builder()
    outputs = builder.call_dict("multi", builder.inputs)
    result = builder.call_dict("fixed", {"xx": outputs["b"], "yy": outputs[right]})
    outputs["b"], outputs[right] = result["xx"], result["yy"]
    record = builder.finalize(outputs)
    assert verify_program_record(record) == ()
    right_output = outputs[right]
    assert isinstance(right_output, RegisterBinding)
    assert right_output.register.dtype == fixed_point_register_type(8, 8)
    calls = tuple(iter_component_calls(record))
    assert [
        record.functions[0].blocks[0].operations[i].callee for i, _, _ in calls
    ] == ["multi", "fixed"]
    assert len(calls[1][1]) == len(calls[1][2]) == 2
    copied = copy_component_program(record)
    assert verify_program_record(copied) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, reference)
    assert _native_trace(copied) == _source_trace(SourceCalls, reference.copy())
    # Both leaves are declarations, so execution has no invented body.
    with pytest.raises(ValueError, match="requires component bodies"):
        lower_register_program(record)
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(
            record,
            (
                (0, 1, 2, 3),
                tuple(range(4, 12)),
                tuple(range(12, 20)),
                tuple(range(20, 28)),
            ),
        )


@pytest.mark.parametrize("mismatch", ["bounded", "sign"])
def test_original_multityped_errors_keep_consumption_prefix(mismatch: str) -> None:
    from qualtran import BloqBuilder, BloqError, QFxp
    from qualtran.bloqs.for_testing.many_registers import (
        TestMultiTypedRegister,
        TestQFxp,
    )

    source, source_inputs = BloqBuilder.from_signature(
        TestMultiTypedRegister().signature, add_registers_allowed=True
    )
    source_extra = (
        source.add_register_from_dtype("e", QFxp(8, 7, True))
        if mismatch == "sign"
        else None
    )
    source_outputs = source.add_d(TestMultiTypedRegister(), **source_inputs)
    if mismatch == "sign":
        for right in ("c", "d"):
            fixed = source.add_d(
                TestQFxp(), xx=source_outputs["b"], yy=source_outputs[right]
            )
            source_outputs["b"], source_outputs[right] = fixed["xx"], fixed["yy"]
    source_first = source_outputs["b"]
    source_second = source_outputs["a"] if mismatch == "bounded" else source_extra
    before = len(source._cxns)
    with pytest.raises(BloqError, match="not consistent"):
        source.add_d(TestQFxp(), xx=source_first, yy=source_second)
    assert source_first not in source._available
    assert source_second not in source._available
    assert len(source._cxns) - before == 2
    builder = _typed_builder()
    if mismatch == "sign":
        signature = Signature(
            (*builder._registers, Register("e", fixed_point_register_type(8, 7, True)))
        )
        builder = ComponentProgramBuilder(signature, tuple(builder._functions.values()))
    outputs = builder.call_dict(
        "multi", {name: builder.inputs[name] for name in "abcd"}
    )
    if mismatch == "sign":
        for right in ("c", "d"):
            result = builder.call_dict(
                "fixed", {"xx": outputs["b"], "yy": outputs[right]}
            )
            outputs["b"], outputs[right] = result["xx"], result["yy"]
    first = outputs["b"]
    second = outputs["a"] if mismatch == "bounded" else builder.inputs["e"]
    with pytest.raises(RegisterBindingError, match="not consistent"):
        builder.call_dict("fixed", {"xx": first, "yy": second})
    assert first not in builder._available and second not in builder._available
    assert len(builder._operations[-1].operands) == 2
    assert builder._operations[-1].results == ()


@pytest.mark.parametrize("shape", [(), (2,), (0,)])
@pytest.mark.parametrize("kind", ["opaque", "uint", "signed", "bounded", "fixed"])
def test_symbolic_width_call_copy_and_expansion(
    kind: str, shape: tuple[int, ...]
) -> None:
    from qualtran import BloqBuilder, BQUInt, QAny, QFxp, QInt, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    n = sympy.Symbol("n", integer=True, positive=True)
    dtype = {
        "opaque": RegisterType("opaque", n),
        "uint": RegisterType("uint", n),
        "signed": RegisterType("signed", n),
        "bounded": bounded_unsigned_register_type(n, 2),
        "fixed": fixed_point_register_type(n, 1),
    }[kind]
    signature = Signature((Register("q", dtype, shape),))
    source_dtype = {
        "opaque": QAny(n),
        "uint": QUInt(n),
        "signed": QInt(n),
        "bounded": BQUInt(n, 2),
        "fixed": QFxp(n, 1),
    }[kind]
    source_signature = SourceSignature(
        (SourceRegister("q", source_dtype, shape=shape),)
    )
    inner, inner_inputs = BloqBuilder.from_signature(source_signature)
    source_identity = inner.finalize(**inner_inputs)
    source, source_inputs = BloqBuilder.from_signature(source_signature)
    reference = source.finalize(**source.add_d(source_identity, **source_inputs))
    values = tuple(
        ValueRef("entry", i, "register") for i, _ in enumerate(np.ndindex(shape))
    )
    leaf = FunctionRecord(
        "identity",
        (BlockRecord("entry", values, return_values=values),),
        registers=tuple(signature),
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    assert verify_program_record(record) == ()
    assert len(tuple(iter_component_calls(record))) == 1
    assert _native_trace(record) == _source_trace(SourceCalls, reference)
    copied = copy_component_program(record)
    assert copied.functions[0].registers == tuple(signature)
    expanded, _ = expand_component_calls(record)
    assert expanded.functions[0].blocks[0].operations == ()
    assert (
        expanded.functions[0].blocks[0].return_values
        == expanded.functions[0].blocks[0].arguments
    )
    for graph in (record, copied, expanded):
        with pytest.raises(ValueError, match="unsupported executable"):
            lower_register_program(graph)
        # Keep a physical carrier for empty arrays. Width validation of a
        # nonempty symbolic port happens before its supplied layout is consumed.
        inputs = tuple((i,) for i in range(len(values)))
        with pytest.raises(ValueError, match="unsupported executable"):
            bind_register_program(graph, inputs, clean=(len(values),))


def test_physical_admission_checks_uncalled_empty_symbolic_definition() -> None:
    builder = ComponentProgramBuilder(
        Signature((Register("q", RegisterType("uint", 1)),))
    )
    record = builder.finalize(builder.inputs)
    unreachable = FunctionRecord(
        "unreachable",
        (BlockRecord("entry", (), return_values=()),),
        registers=(Register("empty", RegisterType("uint", sympy.Symbol("n")), (0,)),),
    )
    record = replace(record, functions=(*record.functions, unreachable))
    assert verify_program_record(record) == ()
    for lower in (
        lower_register_program,
        lambda graph: bind_register_program(graph, ((0,),)),
    ):
        with pytest.raises(ValueError, match="unsupported executable"):
            lower(record)


def test_fixed_point_split_join_and_symbolic_allocation_keep_metadata() -> None:
    dtype = fixed_point_register_type(3, 1, True)
    signature = Signature((Register("q", dtype),))
    builder = ComponentProgramBuilder(signature)
    value = builder.inputs["q"]
    assert isinstance(value, RegisterBinding)
    bits = builder.split(value)
    output = builder.join(bits[::-1], dtype=dtype)
    record = builder.finalize({"q": output})
    assert verify_program_record(record) == ()
    assert output.register.dtype == dtype
    assert verify_program_record(lower_register_program(record)) == ()
    bound = bind_register_program(record, ((0, 1, 2),))
    assert bound.outputs == ((2, 1, 0),) and bound.circuit.instructions == ()
    n = sympy.Symbol("n", positive=True, integer=True)
    builder = ComponentProgramBuilder(Signature(()))
    work = builder.allocate(dtype=fixed_point_register_type(n, 1))
    with pytest.raises(ValueError, match="concrete logical"):
        builder.split(work)
    builder.free(work)
    graph = builder.finalize({})
    assert verify_program_record(graph) == ()
    assert len(tuple(iter_component_calls(graph))) == 2
    with pytest.raises(ValueError, match="unsupported executable"):
        bind_register_program(graph, (), clean=(0,))
