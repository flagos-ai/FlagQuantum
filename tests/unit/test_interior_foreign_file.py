"""Match interior allocation separately in transport, tensor and value protocols."""

from collections.abc import Mapping
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_bit_binding import (
    allocate_classical_value,
    free_classical_value,
)
from flagquantum.simulation.classical_swap import swap_classical_values
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.integer_encoding import integer_from_bits
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.interior_workspace_fixture import interior_workspace_fixture
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_original(source: Any, depth: int) -> ProgramRecord:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    reconstructed = bloqs_from_proto(proto)[0]
    if type(source.n) is int:
        assert reconstructed == source
    else:
        # Original argument sympify reconstructs the name without assumptions.
        assert str(reconstructed.n) == str(source.n) and reconstructed != source

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert name == "main" and registers is not None and set(arguments) == {"n"}
        width = registers[0].bitsize
        assert arguments["n"] == width == registers[1].bitsize
        return interior_workspace_fixture(width)

    program = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    if type(source.n) is int:
        assert program.functions[0].registers == tuple(
            native_signature(source.signature)
        )
    return program


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_foreign_interior_root_preserves_source_primitives_and_work_ownership(
    width: int,
    depth: int,
) -> None:
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = InteriorAlloc(width)
    record = resolve_original(source, depth)
    functions = {function.name: function for function in record.functions}
    original_calls = sorted(
        source.decompose_bloq().bloq_instances, key=lambda node: node.i
    )
    for operation, original in zip(
        record.functions[0].blocks[0].operations, original_calls, strict=True
    ):
        assert functions[operation.callee or ""].registers == tuple(
            native_signature(original.bloq.signature)
        )
    x = tuple(reversed(range(width)))
    y, work = tuple(range(width, 2 * width)), tuple(range(2 * width, 3 * width))
    bound = bind_register_program(record, (x, y), clean=work)
    gates, lifetimes = source_workspace_gates(source, {"x": x, "y": y}, work, ())
    assert_source(bound.circuit.instructions, gates)
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == list(
        lifetimes
    )
    assert bound.outputs == (x, y)
    assert get_cost_value(source, QubitCount()) == 3 * width
    assert get_cost_value(source, QECGatesCost()).clifford == 2 * width
    assert len(gates) == 2 * width and all(gate.name == "swap" for gate in gates)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_all_small_classical_inputs_match_source_cleanup_success_or_failure(
    width: int,
) -> None:
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc

    source = InteriorAlloc(width)
    record = interior_workspace_fixture(width)
    order = tuple(position for position, _, _ in iter_component_calls(record))

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        opcode = function.blocks[0].operations[0].opcode
        if opcode == "allocate_register":
            return {
                "reg": allocate_classical_value(
                    width, lambda bits: integer_from_bits(bits, width)
                )
            }
        if opcode == "free_register":
            return free_classical_value(values["reg"])
        assert function.name == "swap_word"
        x, y = swap_classical_values(values["x"], values["y"])
        return {"x": x, "y": y}

    for x, y in product(range(1 << width), repeat=2):
        values: dict[str, object] = {"x": x, "y": y}
        state = ComponentClassicalState(record, order, values, action)
        if y:
            with pytest.raises(ValueError, match="non-zero register"):
                source.call_classically(x=x, y=y)
            with pytest.raises(ValueError, match="nonzero classical register"):
                state.simulate()
            assert state.last_position == 3
        else:
            assert (
                state.simulate()
                == dict(zip(("x", "y"), source.call_classically(x=x, y=y), strict=True))
                == {"x": 0, "y": x}
            )
        assert values == {"x": x, "y": y}


@pytest.mark.parametrize("width", (1, 2))
def test_complete_projected_tensor_matches_source_on_valid_and_invalid_y_subspaces(
    width: int,
) -> None:
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc

    record = interior_workspace_fixture(width)
    order = tuple(position for position, _, _ in iter_component_calls(record))

    def supply(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.registers is not None
        signature = Signature(function.registers)
        if function.name == "swap_word":
            words = (tuple(range(width)), tuple(range(width, 2 * width)))
            bound = bind_register_program(record, words, entry=function.name)
            matrix = get_unitary(bound.circuit, device="cpu", dtype=torch.complex128)
            tensor = register_tensor_from_unitary(matrix, signature)
        else:
            assert function.blocks[0].operations[0].opcode in {
                "allocate_register",
                "free_register",
            }
            tensor = torch.zeros(1 << width, dtype=torch.complex128)
            tensor[0] = 1
        return (bind_register_tensor(tensor, signature, incoming, outgoing),)

    actual = contract_component_tensors(
        record, order, supply, torch.empty((), dtype=torch.complex128)
    )
    expected = torch.as_tensor(
        InteriorAlloc(width).tensor_contract(), dtype=torch.complex128
    )
    actual = actual.reshape(expected.shape)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    words = 1 << width
    assert torch.count_nonzero(actual) == words
    assert (
        torch.count_nonzero(
            actual[:, [x * words + y for x in range(words) for y in range(1, words)]]
        )
        == 0
    )


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_symbolic_source_assumption_loss_and_native_physical_refusal(
    depth: int,
) -> None:
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc

    width = sympy.Symbol("n", positive=True, integer=True)
    source = InteriorAlloc(width)
    direct = interior_workspace_fixture(width)
    assert direct.functions[0].registers is not None
    for native, original in zip(
        direct.functions[0].registers, source.signature, strict=True
    ):
        assert native.dtype.encoding == "opaque" and native.bitsize is width
        assert (native.name, native.bitsize, native.shape, native.side.name) == (
            original.name,
            original.bitsize,
            original.shape,
            original.side.name,
        )
    record = resolve_original(source, depth)
    main = record.functions[0]
    assert len(main.blocks[0].operations) == 4
    assert (
        next(
            function for function in record.functions if function.name == "swap_word"
        ).blocks
        == ()
    )
    assert tuple(
        operation.callee == "swap_word" for operation in main.blocks[0].operations
    ) == (False, True, True, False)
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(record, ())


def test_source_invalid_width_transport_and_decomposition_dispositions() -> None:
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.serialization.bloq import bloqs_to_proto

    with pytest.raises(TypeError, match="missing.*x.*y"):
        InteriorAlloc(0).decompose_bloq()
    for depth in (1, 2):
        with pytest.raises(TypeError, match="missing.*x.*y"):
            bloqs_to_proto(InteriorAlloc(0), max_depth=depth)
    with pytest.raises(ValueError, match="negative dimensions"):
        bloqs_to_proto(InteriorAlloc(-1), max_depth=2)
