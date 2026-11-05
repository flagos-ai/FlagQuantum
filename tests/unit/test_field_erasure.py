"""Source call-chain parity and every measurement branch of product erasure."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.field_erasure import gf2_product_erasure
from flagquantum.compiler.registers.field_multiplication import gf2_multiplication
from flagquantum.compiler.registers.measurement import ParityPhase, measure_x_clear_ir
from flagquantum.core.classical_dataflow import parse_classical_xor
from flagquantum.core.ir import CircuitIR
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11), (5, 37), (8, 283)])
def test_erasure_matches_unmodified_source_calls(width: int, modulus: int) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Source comparison pins Qualtran v0.7.0")
    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.basic_gates import CZ, Discard, MeasureX
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulMBUC, Parity
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, y, product = [tuple(range(i * width, (i + 1) * width)) for i in range(3)]
    slots = tuple(range(20, 20 + width))
    actual = gf2_product_erasure(x, y, product, slots, modulus=modulus)
    measurements = []
    corrections = []
    parities = []
    source_events = []
    parity_bit = 20 + width

    class Recorder:
        """Bind the real upstream method to labels, without reimplementing its loops."""

        def split(self, register: tuple[int, ...]) -> Any:
            return np.asarray(register)

        def join(self, register: Any, *, dtype: Any) -> tuple[int, ...]:
            return tuple(int(w) for w in register)

        def add(self, bloq: Any, **registers: Any) -> Any:
            if isinstance(bloq, MeasureX):
                wire = int(registers["q"])
                bit = slots[product.index(wire)]
                measurements.append((wire, bit))
                source_events.append(("hmr", (wire,), bit))
                return bit
            if isinstance(bloq, Parity):
                bits = tuple(int(b) for b in registers["x"])
                parities.append(bits)
                source_events.append(("classical_xor", bits, parity_bit))
                return registers["x"], bits
            if isinstance(bloq, Discard):
                return None
            assert bloq.subbloq == CZ()
            control = registers["ctrl"]
            bits = control if isinstance(control, tuple) else (int(control),)
            corrections.append(
                ParityPhase((int(registers["q1"]), int(registers["q2"])), bits)
            )
            source_events.append(
                (
                    "cz",
                    (int(registers["q1"]), int(registers["q2"])),
                    parity_bit if isinstance(control, tuple) else int(control),
                )
            )
            return control, registers["q1"], registers["q2"]

    upstream = GF2MulMBUC(QGF(2, width, Poly.Int(modulus)))
    outputs = upstream.build_composite_bloq(Recorder(), x=x, y=y, result=product)
    assert outputs == {"x": x, "y": y}
    assert actual.measurements == tuple(measurements)
    assert actual.corrections == tuple(corrections)
    assert actual.parities == tuple(parities)
    assert len(parities) == width
    # Normalize source MeasureX+discard to physical HMR; map each source Parity
    # result to the same explicitly reusable classical slot on both sides.
    lowered = actual.lower(parity_bit=parity_bit)
    events = [("hmr", (wire,), bit) for wire, bit in actual.measurements]
    for instruction in lowered[3 * width :]:
        if instruction.name == "classical_xor":
            assignment = parse_classical_xor(instruction)
            events.append(("classical_xor", assignment.sources, assignment.target))
        else:
            ((bit, value),) = instruction.metadata["conditions"]
            assert value == 1
            events.append((instruction.name, instruction.wires, bit))
    assert events == source_events
    assert len(corrections) == width**2
    assert get_cost_value(upstream, QECGatesCost()).clifford == width**2
    emitted = Counter(g.name for g in actual.lower(parity_bit=20 + width))
    assert emitted["cz"] == width**2
    assert emitted["classical_xor"] == width
    assert emitted["measure"] == emitted["h"] == emitted["x"] == width
    assert emitted["ccx"] == emitted["t"] == emitted["tdg"] == 0
    target = lower_reversible_phase(CircuitIR(3 * width, lowered))
    native_counts = Counter(op.opcode for op in target.operations)
    assert native_counts["cz"] == width**2
    assert native_counts["hmr"] == native_counts["bit_store0"] == width
    assert native_counts["bit_invert"] == sum(map(len, parities))


def _compute_erase(width: int, modulus: int) -> CircuitIR:
    x, y, product = [tuple(range(i * width, (i + 1) * width)) for i in range(3)]
    forward = gf2_multiplication(
        x, y, product, tuple(range(3 * width, 4 * width)), modulus=modulus
    )
    erase = gf2_product_erasure(
        x, y, forward.outputs[2], tuple(range(width)), modulus=modulus
    )
    return CircuitIR(4 * width, (*forward.instructions, *erase.lower(parity_bit=width)))


@pytest.mark.parametrize("width,modulus", [(1, 3), (2, 7), (3, 11), (5, 37), (8, 283)])
def test_all_input_pairs_all_branch_phases_and_workspace(
    width: int, modulus: int
) -> None:
    """An exact phase-polynomial proof covers all 2^width measurement branches."""
    np = pytest.importorskip("numpy")
    ir = _compute_erase(width, modulus)
    program = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    assert program == lower_reversible_phase(ir)
    size = 1 << width
    a, b = np.repeat(np.arange(size), size), np.tile(np.arange(size), size)
    states = [
        (
            ((a if w < width else b) >> (width - 1 - w % width)) & 1
            if w < 2 * width
            else np.zeros(size**2, dtype=np.int64)
        )
        for w in range(4 * width)
    ]
    original = [s.copy() for s in states]
    # Each bit of phase_mask is the exponent coefficient of one independent
    # measurement result. A zero mask proves +1 phase for EVERY branch at once.
    phase_mask = np.zeros(size**2, dtype=np.int64)
    classical_masks: dict[int, int] = {}
    for op in program.operations:
        if op.opcode == "hmr":
            assert op.result_bit is not None
            classical_masks[op.result_bit] = 1 << op.result_bit
            phase_mask ^= states[op.wires[0]] << op.result_bit
            states[op.wires[0]] = np.zeros_like(phase_mask)
        elif op.opcode == "bit_store0":
            assert op.result_bit is not None
            classical_masks[op.result_bit] = 0
        elif op.opcode == "bit_invert":
            assert op.result_bit is not None and op.condition_bit is not None
            classical_masks[op.result_bit] ^= classical_masks[op.condition_bit]
        elif op.opcode == "cz":
            assert op.condition_bit is not None
            phase_mask ^= (states[op.wires[0]] & states[op.wires[1]]) * classical_masks[
                op.condition_bit
            ]
        else:
            assert op.opcode in {"cx", "ccx"} and op.condition_bit is None
            *controls, target = op.wires
            active = np.ones_like(phase_mask)
            for wire in controls:
                active &= states[wire]
            states[target] ^= active
    np.testing.assert_array_equal(phase_mask, 0)
    for actual, expected in zip(states, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    # The unnormalized branch amplitude is input_amplitude / sqrt(2^width).
    # This tests nonuniform complex phases without a dense 32-qubit allocation.
    amplitudes = ((a + 1) + 1j * (2 * b - 3)).astype(np.complex128)
    amplitudes /= np.linalg.norm(amplitudes)
    for outcome in range(size):
        signs = np.array(
            [(-1) ** int(mask & outcome).bit_count() for mask in phase_mask]
        )
        np.testing.assert_allclose(
            amplitudes * signs / size**0.5, amplitudes / size**0.5, rtol=0, atol=0
        )


@dataclass(frozen=True)
class ComputeErase:
    width: int
    modulus: int

    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.compose(_compute_erase(self.width, self.modulus))


@pytest.mark.integration
@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11)])
def test_append_dynamic_complex_superposition(width: int, modulus: int) -> None:
    count = 4 * width
    state = torch.zeros(1 << count, dtype=torch.complex128)
    for a in range(1 << width):
        for b in range(1 << width):
            state[((a << width) | b) << (2 * width)] = complex(a + 1, 2 * b - 3)
    state /= state.norm()
    circuit = DynamicCircuit(count, inputs=state, dtype=torch.complex128).append(
        ComputeErase(width, modulus)
    )
    result = run_dynamic(circuit, shots=128, seed=29, strategy="trajectory")
    assert len(set(map(tuple, result.classical_bits.tolist()))) == 1 << width
    torch.testing.assert_close(
        result.final_states.reshape(128, -1), state.expand(128, -1), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("width,modulus", [(3, 11), (8, 283)])
def test_kmx_execution_against_native_reference(
    width: int, modulus: int, tmp_path: Path
) -> None:
    km = pytest.importorskip("kickmix")
    program = lower_reversible_phase(_compute_erase(width, modulus))
    path = tmp_path / "field-erasure.kmx"
    path.write_text(encode_kmx(program))
    native = km.Circuit.from_file(str(path))
    shots = 4096
    all_shots = (1 << shots) - 1
    for a, b in [(0, 0), (1, 1), ((1 << width) - 1, 3)]:
        sim = km.Simulator(batch_size=shots, seed=123)
        expected = []
        for value in (a, b):
            expected.extend(
                all_shots if value >> i & 1 else 0 for i in reversed(range(width))
            )
        for wire, bits in enumerate(expected):
            sim.write_across_shots(km.q(wire), bits)
        sim.do(native)
        assert [
            sim.read_across_shots(km.q(wire), out=int) for wire in range(4 * width)
        ] == expected + [0] * (2 * width)
        assert sim.read_phase_flipped_across_shots(out=int) == 0
        outcomes = set()
        packed = [sim.read_across_shots(km.b(bit), out=int) for bit in range(width)]
        for shot in range(shots):
            outcomes.add(
                sum(((bits >> shot) & 1) << bit for bit, bits in enumerate(packed))
            )
        assert len(outcomes) == 1 << width


def test_erasure_and_shared_primitive_validation() -> None:
    with pytest.raises(ValueError, match="equal"):
        gf2_product_erasure((0, 1), (2, 3), (4,), (0, 1), modulus=7)
    with pytest.raises(ValueError, match="distinct"):
        gf2_product_erasure((0, 1), (2, 3), (4, 5), (0, 0), modulus=7)
    with pytest.raises(ValueError, match="distinct"):
        gf2_product_erasure((0, 1), (2, 3), (4, 5), (0, -1), modulus=7)
    with pytest.raises(ValueError, match="distinct"):
        gf2_product_erasure((0, 1), (2, 3), (4, 5), (0, 1), modulus=7).lower(
            parity_bit=0
        )
    with pytest.raises(ValueError, match="nonnegative"):
        measure_x_clear_ir(0, -1)
    with pytest.raises(ValueError, match="one or two"):
        ParityPhase((1, 1), (0,))
    with pytest.raises(ValueError, match="distinct"):
        ParityPhase((0,), (1, 1))
    assert ParityPhase((0,), ()).lower() == ()
    assert [g.name for g in ParityPhase((0,), (1, 3)).lower(parity_bit=9)] == [
        "classical_xor",
        "z",
    ]
    with pytest.raises(ValueError, match="caller-owned"):
        ParityPhase((0,), (1, 3)).lower()
    with pytest.raises(ValueError, match="one or two"):
        ParityPhase((0, 1, 2), (7, 9))


def test_qroam_consumer_accepts_replaced_measurement_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers import qroam
    from flagquantum.core.ir import Instruction

    arguments = dict(
        width=3,
        data_width=2,
        words=(3, 0, 2, 1),
        measurement_bits=(0, 1, 2, 3),
        blocked=True,
        inverse=True,
    )
    expected = qroam.paired_lookup_ir(**arguments)
    calls = []

    def replacement_measure(wire: int, bit: int) -> tuple[Instruction, ...]:
        calls.append(("measure", wire, bit))
        return (
            Instruction("h", (wire,)),
            Instruction(
                "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": bit}
            ),
            Instruction("x", (wire,), metadata={"conditions": ((bit, 1),)}),
        )

    @dataclass(frozen=True)
    class ReplacementPhase:
        wires: tuple[int, ...]
        bits: tuple[int, ...]

        def lower(self) -> tuple[Instruction, ...]:
            calls.append(("phase", self.wires, self.bits))
            return tuple(
                Instruction(
                    "z" if len(self.wires) == 1 else "cz",
                    self.wires,
                    metadata={"conditions": ((bit, 1),)},
                )
                for bit in self.bits
            )

    monkeypatch.setattr(qroam, "measure_x_clear_ir", replacement_measure)
    monkeypatch.setattr(qroam, "ParityPhase", ReplacementPhase)
    assert qroam.paired_lookup_ir(**arguments).instructions == expected.instructions
    assert sum(call[0] == "measure" for call in calls) == 4
    assert any(call[0] == "phase" for call in calls)
