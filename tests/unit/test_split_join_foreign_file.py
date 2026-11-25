"""Actual foreign Split/Join definitions resolve to native word rebindings."""

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_dtype(kind: str) -> Any:
    from qualtran import (
        BQUInt,
        QAny,
        QBit,
        QFxp,
        QInt,
        QIntOnesComp,
        QMontgomeryUInt,
        QUInt,
    )

    return {
        "bit": QBit(),
        "opaque": QAny(3),
        "unsigned": QUInt(4),
        "signed": QInt(3),
        "fixed": QFxp(3, -2),
        "signed_fixed": QFxp(3, 1, signed=True),
        "signed_fixed_negative": QFxp(3, -2, signed=True),
        "empty_fixed": QFxp(0, 0),
        "empty_fixed_negative": QFxp(0, -2),
        "ones_complement": QIntOnesComp(3),
        "bounded": BQUInt(3, 5),
        "montgomery": QMontgomeryUInt(3),
        "empty": QAny(0),
    }[kind]


def assert_source_binding(source: Any, program: ProgramRecord, reverse: bool) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    width = int(source.dtype.num_qubits)
    wires = tuple(reversed(range(width)))
    bits = tuple((q,) for q in wires)
    inputs = bits if reverse else (wires,)
    bound = bind_register_program(program, inputs, borrowed=() if width else (0,))
    assert bound.outputs == ((wires,) if reverse else bits)
    assert bound.events == () and bound.circuit.instructions == ()
    physical = np.asarray(wires, dtype=object)
    operation, views = source.as_cirq_op(
        None, reg=physical.reshape(width, 1) if reverse else physical
    )
    assert operation is None
    assert tuple(views["reg"].flat) == wires
    assert views["reg"].shape == ((width,) if reverse else (width, 1))
    assert source.as_pl_op(object()) is None
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == width
    state = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(134),
    )
    state /= state.norm()
    expected = state
    if width:
        matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        ).reshape(1 << width, 1 << width)
        order = (*wires, width)
        expected = (
            (
                matrix
                @ state.reshape((2,) * (width + 1))
                .permute(order)
                .reshape(1 << width, 2)
            )
            .reshape((2,) * (width + 1))
            .permute(order)
            .flatten()
        )
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize(
    "kind",
    (
        "bit",
        "opaque",
        "unsigned",
        "signed",
        "fixed",
        "signed_fixed",
        "signed_fixed_negative",
        "empty_fixed",
        "empty_fixed_negative",
        "ones_complement",
        "bounded",
        "montgomery",
        "empty",
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_foreign_dtype_libraries_and_original_factory_use_native_rebindings(
    kind: str, reverse: bool, depth: int
) -> None:
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.bookkeeping.join import _join
    from qualtran.bloqs.bookkeeping.split import _split
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = (
        (_join if reverse else _split).make()
        if kind == "unsigned"
        else (Join if reverse else Split)(original_dtype(kind))
    )
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert (
        serialize_component_library(library) == wire
        and bloqs_from_proto(proto)[0] == source
    )

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None and set(arguments) == {"dtype"}
        dtype = arguments["dtype"]
        assert isinstance(dtype, RegisterType)
        result = (join_word_function if reverse else split_word_function)(name, dtype)
        assert result.registers == registers
        return result

    program = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    assert_source_binding(source, program, reverse)


@pytest.mark.parametrize("fraction", (0, -2))
def test_empty_fixed_split_join_in_an_actual_foreign_phase_context(
    fraction: int,
) -> None:
    import numpy as np
    from qualtran import BloqBuilder, QBit, QFxp
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
        bookkeeping_factories,
    )
    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )
    from flagquantum.simulation.unitary import get_unitary

    dtype = QFxp(0, fraction)
    builder, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("reg", dtype), SourceRegister("s", QBit())))
    )
    bits = builder.add(Split(dtype), reg=inputs["reg"])
    word = builder.add(Join(dtype), reg=bits)
    spectator = builder.add(Hadamard(), q=inputs["s"])
    spectator = builder.add(TGate(), q=spectator)
    source = builder.finalize(reg=word, s=spectator)
    wire = bloqs_to_proto(source, max_depth=1).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    program = resolve_component_program(
        library, 0, bookkeeping_factories() | fixed_gate_factories()
    )
    bound = bind_register_program(program, ((), (0,)))
    assert bound.outputs == ((), (0,)) and bound.events == ()
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == [
        ("h", (0,)),
        ("t", (0,)),
    ]
    actual = get_unitary(
        Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
    )
    expected = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    torch.testing.assert_close(
        actual.to(torch.complex128), expected, atol=2e-15, rtol=0
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == model.t == 1 and model.measurement == 0


@pytest.mark.parametrize("reverse", (False, True))
def test_signed_zero_fixed_storage_keeps_separate_pending_admission(
    reverse: bool,
) -> None:
    import numpy as np
    from qualtran import QFxp
    from qualtran.bloqs.bookkeeping import Join, Split

    source = (Join if reverse else Split)(QFxp(0, -2, signed=True))
    signature = native_signature(source.signature)
    dtype = next(reg.dtype for reg in signature if reg.dtype.encoding == "fixed_point")
    function = (join_word_function if reverse else split_word_function)("view", dtype)
    operation, ports = source.as_cirq_op(
        None, reg=np.empty((0, 1) if reverse else (0,), dtype=object)
    )
    assert operation is None and ports["reg"].size == 0
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bind_register_program(
            ProgramRecord((function,)),
            () if reverse else ((),),
            borrowed=(0,),
            entry="view",
        )


@pytest.mark.parametrize("polynomial", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_binary_field_source_sdk_and_native_bodies_preserve_original_wire_refusal(
    polynomial: bool, reverse: bool
) -> None:
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from flagquantum.simulation.finite_field_polynomial_encoding import (
        FiniteFieldPolynomialEncoding,
    )

    field = FiniteFieldEncoding(2, 2)
    native = (
        FiniteFieldPolynomialEncoding(1, field) if polynomial else field
    ).register_type()
    dtype = QGFPoly(1, QGF(2, 2)) if polynomial else QGF(2, 2)
    source = (Join if reverse else Split)(dtype)
    for depth in (0, 1, 2):
        with pytest.raises(TypeError, match="is not recognized"):
            bloqs_to_proto(source, max_depth=depth)
    function = (join_word_function if reverse else split_word_function)("main", native)
    assert_source_binding(source, ProgramRecord((function,)), reverse)
