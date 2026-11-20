"""Retain complete fixture transport and original executable boundaries."""

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
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from tests.unit.large_register_fixture import large_fixture_ir
from tests.unit.multi_register_fixture import (
    companion_signatures,
    multi_register_fixture,
)
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_large_register_fixture import controlled_source_atom
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("kind", ("multi", "bounded", "fixed", "typed"))
def test_original_multi_register_libraries_resolve_canonical_native_factories(
    depth: int, kind: str
) -> None:
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.for_testing.many_registers import (
        TestBoundedQUInt,
        TestMultiRegister,
        TestMultiTypedRegister,
        TestQFxp,
    )
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source: Any = {
        "multi": TestMultiRegister(),
        "bounded": TestBoundedQUInt(),
        "fixed": TestQFxp(),
        "typed": TestMultiTypedRegister(),
    }[kind]
    if kind == "typed" and depth:
        with pytest.raises(BloqError, match="register dtypes are not consistent"):
            bloqs_to_proto(source, max_depth=depth)
        return
    proto = bloqs_to_proto(source, max_depth=depth)
    raw = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(raw)
    assert serialize_component_library(library) == raw
    assert bloqs_from_proto(proto)[0] == source

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert name == "main" and not arguments
        if kind == "multi":
            result = multi_register_fixture()
            assert result.functions[0].registers == registers
            return result
        signature = companion_signatures()[("bounded", "fixed", "typed").index(kind)]
        assert tuple(signature) == registers
        return FunctionRecord(name, (), registers=registers)

    program = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    if kind != "multi":
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(program, ())
        return
    words = ((11,), (10, 0), (9, 1), (8, 2), (7, 3), (6, 4, 5))
    bound = bind_register_program(program, words)
    assert bound.outputs == words and bound.events == ()
    assert bound.circuit.instructions == (
        Instruction("x", (11,)),
        *(Instruction("cx", word) for word in words[1:5]),
    )
    state = torch.randn(
        4096, dtype=torch.complex128, generator=torch.Generator().manual_seed(936)
    )
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = tuple(wire for word in words for wire in word)
    expected = (matrix @ state.reshape((2,) * 12).permute(order).flatten()).reshape(
        (2,) * 12
    )
    inverse = tuple(order.index(wire) for wire in range(12))
    expected = expected.permute(inverse).flatten()
    circuit = Circuit(12, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(12, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("repetitions", (1, 2))
def test_single_selection_source_ladder_has_no_work_and_exact_complex_action(
    repetitions: int,
) -> None:
    from qualtran.bloqs.for_testing.large_bloq import LargeBloq
    from qualtran.resource_counting import QubitCount, get_cost_value

    source = LargeBloq(1, repetitions)
    gates = large_fixture_ir((1,), 0, repetitions, (), ())
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        expected, lifetimes = source_workspace_gates(
            source,
            {"select": (1,), "target": (0,)},
            (),
            (),
            custom_leaf=controlled_source_atom,
        )
    assert_source(gates, expected)
    assert lifetimes == () and gates == (Instruction("cx", (1, 0)),) * repetitions
    state = torch.tensor([1 + 2j, 3 - 1j, -2 + 4j, 2 + 1j], dtype=torch.complex128)
    state /= state.norm()
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected_state = (
        (matrix @ state.reshape(2, 2).T.flatten()).reshape(2, 2).T.flatten()
    )
    circuit = Circuit(2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(2, gates))
    torch.testing.assert_close(
        circuit.state().flatten(), expected_state, atol=1e-14, rtol=0
    )
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        assert get_cost_value(source, QubitCount()) == 2


def test_large_source_transport_and_invalid_input_failures_are_retained() -> None:
    from qualtran.bloqs.for_testing.large_bloq import LargeBloq
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = LargeBloq(2, 1)
    proto = bloqs_to_proto(source, max_depth=0)
    raw = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(raw)
    assert serialize_component_library(library) == raw
    assert library.entries[0].declaration.registers == tuple(
        native_signature(source.signature)
    )
    with pytest.raises(ValueError, match="Unable to find a Bloq"):
        bloqs_from_proto(proto)
    for depth in (1, 2):
        with pytest.raises(ValueError, match="numpy.uint8"):
            bloqs_to_proto(source, max_depth=depth)
    with pytest.raises(TypeError, match="missing.*select"):
        LargeBloq(0, 0).decompose_bloq()
    with pytest.raises(AssertionError):
        LargeBloq(2, -1).decompose_bloq()
    with pytest.raises(ValueError, match="at least one select"):
        large_fixture_ir((), 0, 0, (), ())
    with pytest.raises(ValueError, match="nonnegative repetitions"):
        large_fixture_ir((0, 1), 2, -1, (), ())
