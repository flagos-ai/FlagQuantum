"""Capture complete atomic and word-swap sources with native presentation."""

from functools import partial
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
)
from flagquantum.compiler.registers.primitives.controlled_swap import controlled_swap_ir
from flagquantum.compiler.registers.primitives.swap import swap_register_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import swap

    for name, blob in (
        ("swap.py", "5c2481f7a1ee0a592d1192999ddf7e6dba8facc3"),
        ("swap_test.py", "9f5d563a9e642e05e46bb4141438686a7b349a08"),
    ):
        raw = (Path(swap.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("kind", ("bit", "word", "controlled_bit", "controlled_word"))
def test_actual_swap_sources_and_owned_round_trip(kind: str) -> None:
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap, TwoBitSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from tests.unit.test_register_subtraction import assert_source

    sources = {
        "bit": TwoBitSwap(),
        "word": Swap(2),
        "controlled_bit": TwoBitCSwap(),
        "controlled_word": CSwap(2),
    }
    source = sources[kind]
    width = 2 if kind.endswith("word") else 1
    controlled = kind.startswith("controlled")
    ctrl_count = int(controlled)
    left = tuple(range(ctrl_count, ctrl_count + width))
    right = tuple(range(ctrl_count + width, ctrl_count + 2 * width))
    logical = 2 * width + ctrl_count
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    native = (
        controlled_swap_ir(0, left, right)
        if controlled
        else swap_register_ir(left, right)
    )
    assert source.adjoint() is source
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    for snapshot in emitter.snapshots:
        restored = CircuitIR.from_json(snapshot.to_json())
        assert_source(restored.instructions, native)
        assert len(restored.instructions) == (17 if controlled else 1) * width
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128), matrix, rtol=0, atol=3e-12
        )
    state = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(1 << (logical + 1))],
        dtype=torch.complex128,
    )
    state /= state.norm()
    # Reverse logical binding, then retain a final entangled spectator.
    wires = tuple(reversed(range(logical)))
    bindings = {"x": tuple(wires[q] for q in left), "y": tuple(wires[q] for q in right)}
    if controlled:
        bindings["ctrl"] = (wires[0],)
    order = (*wires, logical)
    inverse_order = tuple(order.index(q) for q in range(logical + 1))
    expected = (
        (
            matrix
            @ state.reshape((2,) * (logical + 1))
            .permute(order)
            .reshape(1 << logical, 2)
        )
        .reshape((2,) * (logical + 1))
        .permute(inverse_order)
        .flatten()
    )
    circuit = Circuit(logical + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(logical + 1, emitter(bindings, (), (), False)))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=3e-12)
    circuit.append(FieldOperation(logical + 1, emitter(bindings, (), (), True)))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=3e-12)
    model = ControlledSwapCall(width) if controlled else CliffordCall("swap_bit", 2)
    multiplier = 1 if controlled else width
    actual = query_qec_gate_cost(model).asdict()
    assert {
        name: count * multiplier for name, count in actual.items()
    } == get_cost_value(source, QECGatesCost()).asdict()
    assert get_cost_value(source, QubitCount()) == logical


@pytest.mark.parametrize("controlled", (False, True))
def test_original_atomic_swap_pennylane_assertions(controlled: bool) -> None:
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap, TwoBitSwap

    from flagquantum.core.ir import Instruction
    from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

    qml = pytest.importorskip("pennylane")
    source = TwoBitCSwap() if controlled else TwoBitSwap()
    width = 3 if controlled else 2
    ir = CircuitIR(
        width, (Instruction("cswap" if controlled else "swap", tuple(range(width))),)
    )
    script = to_pennylane(ir)
    assert script.operations[0] == source.as_pl_op(wires=list(range(width)))
    assert from_pennylane(script).instructions == ir.instructions
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(script.operations[0]), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=3e-12,
    )


@pytest.mark.parametrize("kind", ("bit", "word", "controlled_bit", "controlled_word"))
def test_full_swap_score_geometry(kind: str) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap, TwoBitSwap
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from flagquantum.drawer.swap_symbols import fredkin_wire_symbol, swap_wire_symbol
    from tests.unit.test_drawer_component_score import assert_source_geometry
    from tests.unit.test_register_tensor import native_signature

    source = {
        "bit": TwoBitSwap(),
        "word": Swap(2),
        "controlled_bit": TwoBitCSwap(),
        "controlled_word": CSwap(2),
    }[kind]
    registers = tuple(native_signature(source.signature))
    names = tuple(reg.name for reg in registers)
    step = RegisterStep(
        str(source), names, names, input_registers=registers, output_registers=registers
    )
    widths = {reg.name: int(reg.bitsize) for reg in registers}
    score = layout_register_score(widths, names, (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    supplier = (
        fredkin_wire_symbol
        if kind == "controlled_bit"
        else partial(swap_wire_symbol, controlled=kind == "controlled_word")
    )
    data = (
        register_score_data(score)
        if kind == "bit"
        else register_score_data(score, register_wire_symbol=supplier)
    )
    assert_source_geometry(data, get_musical_score_data(original), ordered_ports=True)


@pytest.mark.parametrize("width", (0, 1, 3))
@pytest.mark.parametrize("unicode", (False, True))
def test_original_controlled_swap_complete_diagram(width: int, unicode: bool) -> None:
    import cirq
    from qualtran.bloqs.basic_gates.swap import CSwap

    from flagquantum.drawer.swap_symbols import controlled_swap_diagram_symbols

    source = CSwap(width)
    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT.with_args(
        use_unicode_characters=unicode
    )
    assert (
        controlled_swap_diagram_symbols(width, use_unicode=unicode)
        == source._circuit_diagram_info_(args).wire_symbols
    )


def test_source_make_on_inference_and_four_bit_full_operator() -> None:
    import cirq
    from qualtran.bloqs.basic_gates.swap import CSwap

    width = 4
    qubits = cirq.LineQubit.range(2 * width + 1)
    operation = CSwap.make_on(
        ctrl=qubits[:1], x=qubits[1 : width + 1], y=qubits[width + 1 :]
    )
    assert operation.gate == CSwap(width)
    native = CircuitIR(
        2 * width + 1,
        controlled_swap_ir(
            0, tuple(range(1, width + 1)), tuple(range(width + 1, 2 * width + 1))
        ),
    )
    torch.testing.assert_close(
        get_unitary(native, dtype=torch.complex128),
        torch.as_tensor(CSwap(width).tensor_contract()),
        rtol=0,
        atol=4e-12,
    )


def test_source_unknown_swap_labels_and_symbolic_diagram_refusal() -> None:
    import cirq
    from qualtran import QBit
    from qualtran import Register as ForeignRegister
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap
    from sympy import Integer, Symbol

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.register_score import RegisterStep
    from flagquantum.drawer.swap_symbols import (
        controlled_swap_diagram_symbols,
        fredkin_wire_symbol,
        swap_wire_symbol,
    )

    reg = Register("unknown", opaque_register_type(1))
    foreign = ForeignRegister("unknown", QBit())
    step = RegisterStep("swap", ("unknown",), ("unknown",))
    with pytest.raises(ValueError, match="Bad register name"):
        swap_wire_symbol(step, reg, ())
    with pytest.raises(ValueError, match="Bad register name"):
        Swap(1).wire_symbol(foreign)
    assert swap_wire_symbol(step, reg, (), controlled=True) == (
        "Circle",
        {"filled": True},
    )
    assert type(CSwap(1).wire_symbol(foreign)).__name__ == "Circle"
    assert fredkin_wire_symbol(step, reg, ()) == ("TextBox", {"text": "×"})
    assert TwoBitCSwap().wire_symbol(foreign).text == "×"
    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    for value in (Integer(2), Symbol("n")):
        source = CSwap(value)
        if value.is_number:
            assert (
                controlled_swap_diagram_symbols(source.bitsize)
                == source._circuit_diagram_info_(args).wire_symbols
            )
        else:
            with pytest.raises(TypeError):
                controlled_swap_diagram_symbols(source.bitsize)
            with pytest.raises(TypeError):
                source._circuit_diagram_info_(args)
