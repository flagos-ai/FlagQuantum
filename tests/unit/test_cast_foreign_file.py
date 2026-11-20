"""Resolve actual Cast libraries through existing native storage owners."""

from collections.abc import Mapping
from math import prod

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
    from qualtran import QAny, QFxp
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.bloqs.bookkeeping.cast import _cast
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
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


def test_classical_word_source_view_remains_an_explicit_native_gap() -> None:
    from qualtran import CInt, CUInt
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.core.ir.program.register_types import (
        classical_signed_register_type,
        classical_unsigned_register_type,
    )

    source = Cast(CUInt(4), CInt(4))
    assert [source.on_classical_vals(reg=i)["reg"] for i in range(16)] == list(
        range(8)
    ) + list(range(-8, 0))
    value = object()
    assert source.as_cirq_op(None, reg=value) == (None, {"reg": value})
    for depth in (0, 1, 2):
        with pytest.raises(TypeError, match="CUInt.*is not recognized"):
            bloqs_to_proto(source, max_depth=depth)
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        cast_word_function(
            "classical_view",
            classical_unsigned_register_type(4),
            classical_signed_register_type(4),
        )
