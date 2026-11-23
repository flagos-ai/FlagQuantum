"""Replay original builder lifetime and complete consumer observations."""

import json
from copy import copy
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.formats.kmx_block_marks import decode_kmx_block_marks
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.layout.wire_allocation import WireAllocation
from flagquantum.compiler.registers.primitives.broadcast_binding import broadcast_x_ir
from flagquantum.compiler.registers.primitives.broadcast_ccx import broadcast_ccx_ir
from flagquantum.compiler.registers.primitives.broadcast_swap import broadcast_swap_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.drawer.reversible_phase_analysis import (
    render_reversible_phase_analysis_svg,
)
from flagquantum.ecosystem._phase_angle import phase_angle_from_half_turns
from tests.unit.test_physical_circuit_binding import observe

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_builder_binding.json").read_text()
)


def _action(builder: KmxCircuitBuilder, action: list[Any]) -> list[str] | None:
    match action:
        case ["max", value]:
            builder.set_max_qubits(float("inf") if value == "inf" else value)
        case ["quantum", count, name]:
            return [
                x.diagnostic_text for x in builder.create_quantum_register(count, name)
            ]
        case ["classical", count, name]:
            return [
                x.diagnostic_text
                for x in builder.create_classical_register(count, name)
            ]
        case ["alloc_q", count]:
            return [x.diagnostic_text for x in builder.alloc_qubits(count)]
        case ["alloc_b", count]:
            return [x.diagnostic_text for x in builder.alloc_bits(count)]
        case ["free", operands]:
            values = tuple(
                (
                    OperandIdentity(OperandKind.QUANTUM, value)
                    if kind == "q"
                    else (
                        OperandIdentity(OperandKind.CLASSICAL, value)
                        if kind == "b"
                        else bool(value)
                    )
                )
                for kind, value in operands
            )
            builder.free(values)
        case _:
            raise AssertionError(action)
    return None


@pytest.mark.parametrize("trace", _DATA["traces"])
def test_complete_allocation_transactions(trace: dict[str, Any]) -> None:
    builder = KmxCircuitBuilder()
    for action, expected in zip(trace["actions"], trace["observations"], strict=True):
        if expected["error"] is not None:
            with pytest.raises(ValueError):
                _action(builder, action)
        else:
            assert _action(builder, action) == expected["result"]
        assert str(builder.max_qubits) == expected["max"]
        assert str(builder.num_free_qubits) == expected["free"]
        assert builder.num_allocated_qubits == expected["allocated"]
        assert observe(builder.freeze()) == expected["circuit"]


def _append_gates(builder: KmxCircuitBuilder, gates: tuple[Instruction, ...]) -> None:
    width = max((wire for gate in gates for wire in gate.wires), default=0) + 1
    program = lower_reversible_phase(CircuitIR(width, gates))
    builder.append_kmx_text("\n".join(encode_kmx(program).splitlines()[1:]))


def test_reusable_nested_failed_and_out_of_order_marks() -> None:
    builder = KmxCircuitBuilder()
    q = builder.create_quantum_register(4, "data")
    reused = builder.mark("reused")
    for _ in range(2):
        with reused as entered:
            assert entered is reused
            _append_gates(builder, broadcast_ccx_ir(q[0], q[1], q[2]))
    with builder.mark("outer"), builder.mark("inner <0 & 1>"):
        _append_gates(builder, broadcast_ccx_ir(q[1], q[2], q[3]))
    with builder.mark(""):
        builder.append_kmx_text("Z_POW q0 0.1")
    with (
        pytest.raises(RuntimeError, match="body failure"),
        builder.mark('failed "block"\n'),
    ):
        _append_gates(builder, broadcast_x_ir(q[3]))
        raise RuntimeError("body failure")
    first, second = builder.mark("first"), builder.mark("second")
    first.__enter__()
    second.__enter__()
    assert first.__exit__(None, None, None) is False
    assert second.__exit__(None, None, None) is False
    builder.append_kmx_text("DEBUG_PRINT q0\nDEBUG_PRINT q3\nDEBUG_PRINT")
    assert observe(builder.freeze()) == _DATA["marks"]


def test_original_broadcast_swap_and_exact_phase_consumers() -> None:
    import kickmix as km

    source = km.CircuitBuilder()
    native = KmxCircuitBuilder()
    left = native.create_quantum_register(5, "left")
    right = native.create_quantum_register(5, "right")
    source_left = source.create_quantum_register(5, "left")
    source_right = source.create_quantum_register(5, "right")
    source.swap(source_left, source_right)
    _append_gates(native, broadcast_swap_ir(left, right))
    for targets, angle in (((0,), 0.1), ((1,), "0.1"), ((2, 3), 2.125), ((4,), -0.125)):
        source.z_pow([km.q(wire) for wire in targets], angle)
        converted = phase_angle_from_half_turns(angle)
        for wire in targets:
            native.append_kmx_text(f"Z_POW q{wire} {converted.to_decimal_half_turns()}")
    assert str(source.finish_circuit()) == native.freeze().to_kmx()


def test_original_reusable_marks_feed_existing_flame_renderer() -> None:
    from xml.etree.ElementTree import fromstring

    builder = KmxCircuitBuilder()
    q = builder.create_quantum_register(4, "q")
    reused = builder.mark("reused")
    for _ in range(2):
        with reused:
            _append_gates(builder, broadcast_ccx_ir(q[0], q[1], q[2]))
    with builder.mark("outer"), builder.mark("inner <0 & 1>"):
        _append_gates(builder, broadcast_ccx_ir(q[1], q[2], q[3]))
    svg = render_reversible_phase_analysis_svg(builder.freeze().to_kmx())
    fromstring(svg)
    for expected in (
        ">reused (x2)</text>",
        ">outer</text>",
        ">inner &lt;0 &amp; 1&gt;</text>",
        ">4 qubits, 3 CCZ states",
    ):
        assert expected in svg


def test_source_integer_protocols_and_budget_float32_boundary() -> None:
    import kickmix as km

    class Index:
        def __index__(self) -> int:
            return 2

    class Int:
        def __int__(self) -> int:
            return 2

    class Integer(int):
        pass

    values: tuple[Any, ...] = (False, True, Integer(2), Index(), Int())
    for value in values:
        source = km.CircuitBuilder()
        native = KmxCircuitBuilder()
        assert [str(x) for x in source.alloc_qubits(value)] == [
            x.diagnostic_text for x in native.alloc_qubits(value)
        ]
    source = km.CircuitBuilder()
    native = KmxCircuitBuilder()
    for limit in (
        float.fromhex("0x1.fffffefffffffp+127"),
        float("nan"),
        float("-inf"),
    ):
        with pytest.raises(ValueError):
            source.set_max_qubits(limit)
        with pytest.raises(ValueError):
            native.set_max_qubits(limit)


def test_lease_copy_clear_and_frozen_values_are_independent() -> None:
    builder = KmxCircuitBuilder()
    builder.set_max_qubits(7)
    register = builder.create_quantum_register(2, "data")
    temporary = builder.alloc_qubits(3)
    builder.free(temporary[1])
    frozen = builder.freeze()
    duplicate = copy(builder)
    assert duplicate.alloc_qubits(2) == builder.alloc_qubits(2)
    builder.free(temporary[0])
    assert duplicate.num_allocated_qubits == 6
    assert builder.num_allocated_qubits == 5
    duplicate.free(temporary[1:])
    with pytest.raises(ValueError):
        duplicate.free(temporary[1:])
    assert builder.num_allocated_qubits == 5
    builder.clear()
    assert builder.num_allocated_qubits == 0
    assert builder.max_qubits == float("inf")
    assert len(builder.freeze().registers) == 0
    assert len(frozen.registers) == 1
    assert [x.value for x in register] == [0, 1]
    assert builder.alloc_qubits(2) == register


def test_budget_refusals_and_conversion_before_partial_release() -> None:
    builder = KmxCircuitBuilder()
    q = builder.alloc_qubits(2)
    for limit in (-1, 1.5, float("nan"), float("-inf"), 1 << 64, 1):
        with pytest.raises(ValueError):
            builder.set_max_qubits(limit)
    invalid_counts: tuple[Any, ...] = (-1, 1.5)
    for invalid_count in invalid_counts:
        with pytest.raises(TypeError):
            builder.alloc_qubits(invalid_count)
        with pytest.raises(TypeError):
            builder.alloc_bits(invalid_count)
    with pytest.raises(ValueError):
        builder.free((q[0], object()))
    assert builder.num_allocated_qubits == 2
    with pytest.raises(ValueError):
        builder.free((q[0], True))
    assert builder.num_allocated_qubits == 1
    assert builder.alloc_qubits(1)[0] == q[0]
    invalid_names: tuple[Any, ...] = (None, 1, object())
    for invalid_name in invalid_names:
        with pytest.raises(TypeError):
            builder.mark(invalid_name)
        with pytest.raises(TypeError):
            builder.create_quantum_register(1, invalid_name)
        with pytest.raises(TypeError):
            builder.create_classical_register(1, invalid_name)
    assert builder.num_allocated_qubits == 2


def test_allocator_limit_and_no_emitted_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.compiler.registers.layout import wire_allocation as owner

    monkeypatch.setattr(owner, "MAX_NUM_QUBITS", 4)
    storage = WireAllocation()
    assert storage.allocate(2) == (0, 1)
    assert storage.reserve(2) == (2, 3)
    with pytest.raises(ValueError, match="ID limit"):
        storage.allocate(1)
    storage.free(0)
    assert storage.allocate(1) == (0,)
    with pytest.raises(TypeError):
        storage.reserve(-1)
    assert storage.allocated_count == 4
    builder = KmxCircuitBuilder()
    q = builder.alloc_qubits(2)
    builder.append_kmx_text("X q0")
    builder.free(q)
    assert [x.value for x in builder.alloc_qubits(2)] == [1, 0]
    assert builder.freeze().to_kmx() == "X q0"


def _built_instructions(program: ReversiblePhaseProgram) -> tuple[Instruction, ...]:
    circuit = read_kmx_circuit(encode_kmx(program))
    builder = KmxCircuitBuilder()
    builder.replace_registers(circuit.registers)
    builder.append(KmxCircuit((), circuit.records[:1]))
    with builder.mark("replaced construction"):
        builder.append(KmxCircuit((), circuit.records[1:]))
    restored, calls = decode_kmx_block_marks(builder.freeze().to_kmx())
    assert len(calls) == 1
    assert restored == program
    return bind_reversible_phase_execution(restored)


def test_builder_replacement_uses_normal_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.algorithms import unary_address_traversal as consumer

    monkeypatch.setattr(
        consumer, "bind_reversible_phase_execution", _built_instructions
    )
    consumer.main()


@pytest.mark.parametrize("n", (0, 1, 2))
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("park", (False, True))
def test_builder_replacement_preserves_all_complex_branches(
    n: int, controlled: bool, park: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit import test_kickmix_unary_iteration as consumer

    monkeypatch.setattr(
        consumer, "bind_reversible_phase_execution", _built_instructions
    )
    consumer.test_all_reset_and_erasure_branches_preserve_complex_amplitudes(
        n, controlled, park
    )
