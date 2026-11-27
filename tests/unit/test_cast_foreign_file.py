"""Resolve actual Cast libraries through existing native storage owners."""

from collections.abc import Mapping
from math import prod
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
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("example", (True, False))
def test_actual_factory_and_shaped_library_resolve_existing_executable_view(
    depth: int, example: bool
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran._infra.data_types import QAny, QFxp
    from qualtran.bloqs.bookkeeping.cast import Cast, _cast
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = _cast.make() if example else Cast(QAny(4), QFxp(4, 2), shape=(2,))
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source
    assert len(library.entries) == 1

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None
        assert set(arguments) == {
            "inp_dtype",
            "out_dtype",
            "shape",
            "allow_quantum_to_classical",
        }
        inp, out = arguments["inp_dtype"], arguments["out_dtype"]
        assert isinstance(inp, RegisterType) and isinstance(out, RegisterType)
        shape = arguments["shape"]
        assert np.asarray(shape).tolist() == list(registers[0].shape)
        assert not arguments["allow_quantum_to_classical"]
        result = cast_word_function(name, inp, out, registers[0].shape)
        assert result.registers == registers
        return result

    program = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    width = source.inp_dtype.num_bits
    count = prod(source.shape)
    words = tuple(
        tuple(reversed(range(i * width, (i + 1) * width))) for i in range(count)
    )
    bound = bind_register_program(program, words)
    assert (
        bound.outputs == words
        and bound.events == ()
        and bound.circuit.instructions == ()
    )
    array = np.asarray(words, dtype=object).reshape((*source.shape, width))
    operation, outputs = source.as_cirq_op(None, reg=array)
    assert operation is None and outputs["reg"] is array
    assert source.as_pl_op(object()) is None
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == width * count
    if not example:
        state = torch.randn(
            1 << (width * count + 1),
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(814),
        )
        state /= state.norm()
        circuit = Circuit(width * count + 1, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(width * count + 1, bound.circuit.instructions))
        torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=0)


def test_classical_word_source_view_binds_with_original_transport_failure_retained() -> (
    None
):
    from qualtran._infra.data_types import CInt, CUInt
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.core.ir.program.register_types import (
        classical_signed_register_type,
        classical_unsigned_register_type,
    )

    source = Cast(CUInt(4), CInt(4))
    assert [source.on_classical_vals(reg=i)["reg"] for i in range(16)] == list(
        range(8)
    ) + list(range(-8, 0))
    value: Any = object()
    assert source.as_cirq_op(None, reg=value) == (None, {"reg": value})
    for depth in (0, 1, 2):
        with pytest.raises(TypeError, match="CUInt.*is not recognized"):
            bloqs_to_proto(source, max_depth=depth)
    program = cast_word_function(
        "main", classical_unsigned_register_type(4), classical_signed_register_type(4)
    )
    from flagquantum.core.ir.program import ProgramRecord

    for raw in range(16):
        bound = bind_register_program(
            ProgramRecord((program,)), ((),), classical_inputs=(raw,), borrowed=(0,)
        )
        assert bound.classical_outputs == ((0, 1, 2, 3),)
        assert len(bound.circuit.instructions) == 4  # Initial word, no cast gate.
        assert bound.events == ()


@pytest.mark.parametrize("reverse", (False, True))
def test_classical_source_callbacks_keep_slots_and_zero_model_cost(
    reverse: bool,
) -> None:
    from qualtran._infra.data_types import (
        CFxp,
        CInt,
        CIntOnesComp,
        CMontgomeryUInt,
        CUInt,
    )
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.core.ir.program.register_types import (
        classical_fixed_point_register_type,
        classical_montgomery_register_type,
        classical_ones_complement_register_type,
        classical_signed_register_type,
        classical_unsigned_register_type,
    )

    pairs: tuple[tuple[Any, RegisterType], ...] = (
        (CInt(3), classical_signed_register_type(3)),
        (CIntOnesComp(3), classical_ones_complement_register_type(3)),
        (CMontgomeryUInt(3, 5), classical_montgomery_register_type(3, 5)),
        (CFxp(3, 1), classical_fixed_point_register_type(3, 1)),
        (CFxp(3, 1, signed=True), classical_fixed_point_register_type(3, 1, True)),
    )
    for foreign, dtype in pairs:
        inp, out = (foreign, CUInt(3)) if reverse else (CUInt(3), foreign)
        left, right = (
            (dtype, classical_unsigned_register_type(3))
            if reverse
            else (classical_unsigned_register_type(3), dtype)
        )
        source = Cast(inp, out, shape=(2,))
        native = cast_word_function("main", left, right, (2,))
        slots: Any = object()
        assert source.as_cirq_op(None, reg=slots) == (None, {"reg": slots})
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == 0
        for raw in range(8):
            bits = [(raw >> i) & 1 for i in (2, 1, 0)]
            value = inp.from_bits(bits)
            result = source.on_classical_vals(reg=value)["reg"]
            assert result == out.from_bits(inp.to_bits(value))
            # Ones-complement negative zero can canonicalize numerically;
            # the physical hook and native cast still preserve every raw slot.
            stored = raw if left.encoding != "signed" or raw < 4 else raw - 8
            if reverse and left.encoding == "ones_complement":
                stored = -(7 ^ raw) if raw & 4 else raw
            if (
                reverse
                and left.encoding == "fixed_point"
                and dict(left.parameters)["signed"]
                and raw >= 4
            ):
                stored = raw - 8
            bound = bind_register_program(
                ProgramRecord((native,)),
                ((), ()),
                classical_inputs=(stored, stored),
                borrowed=(0,),
            )
            assert bound.classical_outputs == ((0, 1, 2), (3, 4, 5))
            assert len(bound.circuit.instructions) == 6
            assert bound.events == ()
