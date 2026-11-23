"""Close complete swap factories through owned typed word and atomic bodies."""

from collections import Counter

import pytest
import sympy
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_exchange_factories import exchange_factories
from flagquantum.compiler.registers.primitives.swap_component import (
    SwapDeclaration,
    swap_word_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import (
    verify_program_record,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "example",
    (
        "_swap_bit",
        "_cswap_bit",
        "_swap",
        "_swap_small",
        "_swap_large",
        "_cswap",
        "_cswap_small",
        "_cswap_large",
    ),
)
def test_all_original_factories_actual_libraries_and_native_resolution(
    example: str,
) -> None:
    from qualtran.bloqs.basic_gates import swap
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = getattr(swap, example).make()
    symbolic = example in {"_swap", "_cswap"}
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        assert serialize_component_library(deserialize_component_library(wire)) == wire
        # The unchanged source wire stores symbol names without assumptions.
        assert (bloqs_from_proto(proto)[0] == source) is not symbolic
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    resolved = resolve_component_program(library, 0, exchange_factories())
    assert resolved.functions[0].registers == tuple(native_signature(source.signature))
    if symbolic:
        width = sympy.Symbol("n")
        with pytest.raises(ValueError, match="concrete width"):
            SwapDeclaration(width, example == "_cswap").program()
        return
    signature = native_signature(source.signature)
    offset = 0
    ports = []
    for register in signature:
        assert isinstance(register.bitsize, int)
        ports.append(tuple(range(offset, offset + register.bitsize)))
        offset += register.bitsize
    bound = bind_register_program(resolved, tuple(ports))
    assert bound.outputs == tuple(ports) and not bound.events
    assert verify_program_record(resolved) == ()
    width = 1 if example.endswith("_bit") else source.bitsize
    counts = Counter(g.name for g in bound.circuit.instructions)
    if example.startswith("_cswap"):
        assert counts == {
            "cx": 8 * width,
            "h": 2 * width,
            "t": 4 * width,
            "tdg": 3 * width,
        }
    else:
        assert counts == {"swap": width}


@pytest.mark.parametrize("width", (1, 2, 3, 8, 64))
def test_plain_word_exact_original_call_layers_and_primitive_order(width: int) -> None:
    from qualtran.bloqs.basic_gates.swap import Swap, TwoBitSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Swap(width)
    declaration = SwapDeclaration(width)
    graph = declaration.program()
    assert graph == swap_word_program(width)
    assert declaration.signature == native_signature(source.signature)
    assert declaration.direct_calls() == {
        CliffordCall("swap_bit", 2): source.bloq_counts()[TwoBitSwap()]
    }
    compare_layers(graph, source, atomic_leaves(graph))
    left, right = tuple(reversed(range(width))), tuple(range(width, 2 * width))
    bound = bind_register_program(graph, (left, right))
    assert bound.outputs == (left, right) and not bound.events
    assert bound.circuit.instructions == tuple(
        Instruction("swap", pair) for pair in zip(left, right, strict=True)
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == len(bound.circuit.instructions) == width
    assert get_cost_value(source, QubitCount()) == 2 * width


@pytest.mark.parametrize("controlled", (False, True))
def test_symbolic_word_signature_and_original_direct_leaf_model(
    controlled: bool,
) -> None:
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap, TwoBitSwap

    n = sympy.Symbol("n")
    source = CSwap(n) if controlled else Swap(n)
    declaration = SwapDeclaration(n, controlled)
    leaf = ControlledSwapCall(1) if controlled else CliffordCall("swap_bit", 2)
    original_leaf = TwoBitCSwap() if controlled else TwoBitSwap()
    assert declaration.signature == native_signature(source.signature)
    assert declaration.direct_calls() == {leaf: source.bloq_counts()[original_leaf]}
    with pytest.raises(ValueError, match="concrete width"):
        declaration.program()


@pytest.mark.parametrize("width", (1, 2, 3))
def test_plain_stored_append_complete_complex_spectator_and_actual_source_inverse(
    width: int,
) -> None:
    from qualtran.bloqs.basic_gates.swap import Swap

    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

    source = Swap(width)
    n_wires = 2 * width + 1
    physical = tuple(reversed(range(2 * width)))
    ports = (physical[:width], physical[width:])
    bound = bind_register_program(
        swap_word_program(width), ports, borrowed=(2 * width,)
    )
    raw = torch.arange(1, (1 << n_wires) + 1, dtype=torch.float64)
    state = torch.complex(raw, 7 - 3 * raw)
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*physical, 2 * width)
    inverse = tuple(order.index(q) for q in range(n_wires))
    expected = (
        (
            matrix
            @ state.reshape((2,) * n_wires).permute(order).reshape(1 << (2 * width), 2)
        )
        .reshape((2,) * n_wires)
        .permute(inverse)
        .flatten()
    )
    circuit = fq.Circuit(n_wires, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(n_wires, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=3e-12)
    emitter = import_register_emitter(source)
    undo = emitter({"x": ports[0], "y": ports[1]}, (), (), True)
    circuit.append(FieldOperation(n_wires, undo))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=3e-12)


def test_original_zero_and_symbolic_physical_failures_remain_explicit() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap

    for source in (Swap(0), CSwap(0)):
        with pytest.raises(TypeError):
            source.decompose_bloq()
    for controlled in (False, True):
        with pytest.raises(ValueError, match="positive concrete"):
            SwapDeclaration(0, controlled).program()
    with pytest.raises(DecomposeTypeError):
        Swap(sympy.Integer(3)).decompose_bloq()
