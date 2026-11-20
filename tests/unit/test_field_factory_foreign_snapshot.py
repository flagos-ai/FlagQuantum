"""Preserve actual field-factory graph aliases at the foreign snapshot boundary."""

import builtins
from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.circuit_binding import (
    restore_output_order_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "factory",
    (
        "_gf2_multiply_by_constant",
        "_gf2_poly_multiply_by_constant",
        "_gf2shiftleft",
        "_gf2shiftright",
    ),
)
@pytest.mark.parametrize("reversed_binding", (False, True))
def test_actual_factory_aliases_complete_complex_action_and_original_adjoint(
    factory: str,
    reversed_binding: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as source
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = getattr(source, factory).make()
    emitter = import_register_emitter(original)
    ((name, width),) = emitter.layout
    assert emitter.requirements() == (0, 0)
    n = width + 1
    initial = torch.arange(1 << n, dtype=torch.float64)
    initial = torch.complex(initial + 1, initial.remainder(7) - 3)
    initial /= initial.norm()
    forward_expected = torch.zeros_like(initial)
    backward_expected = torch.zeros_like(initial)
    wires = tuple(reversed(range(width))) if reversed_binding else tuple(range(width))
    ports = {name: wires}
    streams = tuple(emitter(ports, (), (), adjoint) for adjoint in (False, True))
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix", "galois"}:
            raise AssertionError(f"Captured field factory imported {name}")
        return original_import(name, *args, **kwargs)

    def pack(value: int, spectator: int) -> int:
        return spectator | sum(
            ((value >> (width - 1 - bit)) & 1) << (n - 1 - wire)
            for bit, wire in enumerate(wires)
        )

    for adjoint, direction in enumerate((original, original.adjoint())):
        # Flattening the original typed graph preserves its virtual output aliases.
        raw, outputs = reference_gates(direction, ports, (), repair_adjoint=False)
        permutation = tuple(wires.index(wire) for wire in outputs[name])
        swaps = tuple(
            Instruction(gate.name, tuple(wires[wire] for wire in gate.wires))
            for gate in restore_output_order_ir(permutation)
        )
        expected_stream = [*raw, *((gate.name, gate.wires) for gate in swaps)]
        stream = streams[adjoint]
        assert_gate_streams(
            [(gate.name, gate.wires) for gate in stream if gate.name != "swap"],
            raw,
        )
        assert emitter.binding_swaps[adjoint] == len(swaps) > 0
        assert Counter(gate.name for gate in stream) == {
            "cx": get_cost_value(direction, QECGatesCost()).clifford,
            "swap": len(swaps),
        }
        # Compare the complete state against the unchanged graph, including routing.
        expected_ir = CircuitIR(
            n,
            tuple(Instruction(gate, wires) for gate, wires in expected_stream),
        )
        expected = Circuit(n, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(n, expected_ir.instructions)
        )
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", owned_import)
            restored = CircuitIR.from_json(CircuitIR(n, stream).to_json())
            actual = Circuit(n, inputs=initial, dtype=torch.complex128).append(
                FieldOperation(n, restored.instructions)
            )
        torch.testing.assert_close(actual.state(), expected.state(), atol=2e-12, rtol=0)
        destination = backward_expected if adjoint else forward_expected
        for value in range(1 << width):
            if factory.startswith("_gf2shift"):
                field = original.qgf.gf_type
                result = int(direction.on_classical_vals(**{name: field(value)})[name])
            else:
                # The original linear adjoint emits its forward body. Keep that
                # source defect visible instead of repairing the reference.
                field = original.qgf.gf_type
                result = int(original.on_classical_vals(**{name: field(value)})[name])
            for spectator in (0, 1):
                destination[pack(result, spectator)] = initial[pack(value, spectator)]
        torch.testing.assert_close(
            actual.state().flatten(), destination, atol=2e-12, rtol=0
        )
    round_trip = (
        Circuit(n, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(n, streams[0]))
        .append(FieldOperation(n, streams[1]))
    )
    if factory.startswith("_gf2shift"):
        torch.testing.assert_close(
            round_trip.state().flatten(), initial, atol=2e-12, rtol=0
        )
    else:
        assert not torch.allclose(round_trip.state().flatten(), initial)


@pytest.mark.parametrize(
    "factory", ("_multiplypolybyoneplusxk", "_binarypolynomialmultiplication")
)
def test_actual_polynomial_factories_all_pairs_sparse_complex_work_and_adjoint(
    factory: str,
) -> None:
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as source
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_binary_polynomial import carryless_product, execute_sparse

    original = getattr(source, factory).make()
    emitter = import_register_emitter(original)
    work, feedback = emitter.requirements()
    assert (work, feedback) == (1, 0)
    logical = sum(width for _, width in emitter.layout)
    ports: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, width in emitter.layout:
        ports[name] = tuple(range(offset, offset + width))
        offset += width
    for adjoint, direction in enumerate((original, original.adjoint())):
        raw, outputs = reference_gates(
            direction, ports, (logical,), repair_adjoint=False
        )
        assert outputs == ports
        stream = emitter.snapshots[adjoint].instructions
        assert_gate_streams([(gate.name, gate.wires) for gate in stream], raw)
        assert emitter.binding_swaps[adjoint] == 0
        model = get_cost_value(direction, QECGatesCost())
        assert Counter(gate.name for gate in stream) == {
            "ccx": model.toffoli,
            "cx": model.clifford,
        }

    width, target_width = len(ports["f"]), len(ports["h"])
    pairs = torch.arange(1 << (2 * width), dtype=torch.int64).repeat_interleave(4)
    f, g = pairs >> width, pairs & ((1 << width) - 1)
    selector = torch.arange(len(pairs), dtype=torch.int64) % 4
    target = ((pairs * 19 + 7) ^ ((selector // 2) * ((1 << target_width) - 1))) & (
        (1 << target_width) - 1
    )
    product = carryless_product(f, g, width)
    if factory == "_multiplypolybyoneplusxk":
        product ^= product << int(original.k)

    def pack(
        left: torch.Tensor, right: torch.Tensor, accumulator: torch.Tensor
    ) -> torch.Tensor:
        # Original QBit arrays enumerate polynomial coefficients low degree first.
        result = selector % 2
        for name, value in (("f", left), ("g", right), ("h", accumulator)):
            for bit, wire in enumerate(ports[name]):
                result = result | (((value >> bit) & 1) << (logical + work - wire))
        return result

    initial = pack(f, g, target)
    expected = pack(f, g, target ^ product)
    n = logical + work + 1
    forward, backward = emitter.snapshots
    actual = execute_sparse(forward.instructions, initial, n)
    assert torch.equal(actual, expected)
    assert not torch.any((actual >> 1) & 1)
    assert torch.equal(execute_sparse(backward.instructions, actual, n), initial)
    labels = torch.arange(len(pairs), dtype=torch.float64)
    amplitudes = torch.complex(labels + 1, labels.remainder(31) - 15)
    amplitudes /= amplitudes.norm()
    a, b = actual.argsort(), expected.argsort()
    torch.testing.assert_close(amplitudes[a], amplitudes[b], atol=0, rtol=0)


def test_actual_factory_inventory_and_source_failure_stages() -> None:
    import cirq
    import numpy as np
    from qualtran import BloqExample, DecomposeTypeError
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as source

    names = {
        name for name, value in vars(source).items() if isinstance(value, BloqExample)
    }
    assert names == {
        "_gf16_multiplication",
        "_gf2_multiplication_symbolic",
        "_gf2_multiply_by_constant",
        "_gf2_poly_multiply_by_constant",
        "_multiplypolybyoneplusxk",
        "_binarypolynomialmultiplication",
        "_gf2shiftleft",
        "_gf2shiftright",
        "_gf2mulviakaratsuba",
    }
    symbolic = source._gf2_multiplication_symbolic.make()
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        symbolic.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(symbolic, local_ports=("result",))
    # The original generic wrapper counts only quantum storage but merges the
    # classical RIGHT bit into its operand list. No native importer is involved.
    manager = cirq.SimpleQubitManager()
    quantum = manager.qalloc(1)
    with pytest.raises(ValueError, match="Expected 1 qubits"):
        MeasureX().as_cirq_op(qubit_manager=manager, q=np.array(quantum))
    original = source._gf2mulviakaratsuba.make()
    with pytest.raises(ValueError, match="MeasureX.*Expected 1 qubits"):
        import_register_emitter(original, local_ports=("result",))


def test_actual_karatsuba_factory_forward_graph_aliases_and_retained_source_defect() -> (
    None
):
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as source
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.ecosystem.cirq._component_snapshot import _snapshot

    original = source._gf2mulviakaratsuba.make()
    layout = tuple((reg.name, int(reg.total_bits())) for reg in original.signature)
    captured, binding_swaps = _snapshot(original, layout, local_ports=("result",))
    assert captured.n_wires == 15
    ports = {
        "x": tuple(range(5)),
        "y": tuple(range(5, 10)),
        "result": tuple(range(10, 15)),
    }
    # The original graph declares padding allocations even when no primitive
    # touches them. Reserve those original lifetimes in the independent reader.
    raw, outputs = reference_gates(
        original,
        {"x": ports["x"], "y": ports["y"]},
        tuple(range(10, 20)),
        repair_adjoint=False,
    )
    locations = list(range(captured.n_wires))
    normalized: list[tuple[str, tuple[int, ...]]] = []
    for gate in captured.instructions:
        if gate.name == "swap":
            wire_a, wire_b = gate.wires
            locations[wire_a], locations[wire_b] = locations[wire_b], locations[wire_a]
        else:
            normalized.append((gate.name, tuple(locations[w] for w in gate.wires)))
    assert_gate_streams(normalized, raw)
    assert outputs == {
        name: tuple(locations[w] for w in wires) for name, wires in ports.items()
    }
    model = get_cost_value(original, QECGatesCost())
    assert Counter(gate.name for gate in captured.instructions) == {
        "cx": model.clifford,
        "ccx": model.toffoli,
        "swap": binding_swaps,
    }
    assert binding_swaps == 8

    pairs = torch.arange(1024, dtype=torch.int64).repeat_interleave(2)
    spectators = torch.arange(len(pairs), dtype=torch.int64) % 2
    initial = (pairs << 6) | spectators
    n = captured.n_wires + 1
    permutation = tuple(q for name in ports for q in outputs[name])
    reference = tuple(
        Instruction(name, wires) for name, wires in raw
    ) + restore_output_order_ir(permutation)

    def execute(gates: tuple[Instruction, ...]) -> torch.Tensor:
        result = initial.clone()
        for gate in gates:
            bits = tuple(n - wire - 1 for wire in gate.wires)
            if gate.name == "swap":
                flip = ((result >> bits[0]) ^ (result >> bits[1])) & 1
                result ^= (flip << bits[0]) | (flip << bits[1])
            else:
                assert gate.name in {"cx", "ccx"}
                controls = torch.ones_like(result)
                for bit in bits[:-1]:
                    controls &= (result >> bit) & 1
                result ^= controls << bits[-1]
        return result

    artifact = CircuitIR.from_json(captured.to_json())
    actual, expected = execute(artifact.instructions), execute(reference)
    assert torch.equal(actual, expected)
    labels = torch.arange(len(pairs), dtype=torch.float64)
    amplitudes = torch.complex(labels + 1, labels.remainder(17) - 8)
    amplitudes /= amplitudes.norm()
    a, b = actual.argsort(), expected.argsort()
    torch.testing.assert_close(amplitudes[a], amplitudes[b], rtol=0, atol=0)
    # Frozen SynthesizeLRCircuit.adjoint emits its forward network. This exact
    # source graph still differs from field multiplication for some inputs.
    field = original.qgf.gf_type
    mathematical = torch.tensor(
        [int(field(x) * field(y)) for x in range(32) for y in range(32)],
        dtype=torch.int64,
    ).repeat_interleave(2)
    assert torch.any(((actual >> 1) & 31) != mathematical)
