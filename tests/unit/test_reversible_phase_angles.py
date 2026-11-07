"""Exact conditional phase words survive the compiler and numerical boundary."""

from fractions import Fraction
from itertools import product
from math import pi

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128 as Angle
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Op,
)
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseProgram as Program,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering import reversible_phase_execution as binding
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("exponent", [0, -1, -2, -7, -127, -128])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("first,outer", list(product((False, True), repeat=2)))
def test_exact_words_native_phase_and_entangled_guard_execution(
    exponent: int, negative: bool, first: bool, outer: bool
) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    angle = Angle.from_power_of_two_half_turns(exponent)
    if negative:
        angle = -angle
    program = Program(
        2,
        2,
        (
            Op("push_condition", (), condition_bit=1),
            Op("z_pow", (0,), condition_bit=0, angle=angle),
            Op("pop_condition", ()),
        ),
        (0, 1),
    )
    text = encode_kmx(program)
    assert decode_kmx(text) == program
    assert f"Z_POW q0 {angle.to_decimal_half_turns()} if b0" in text
    sim = km.Simulator(batch_size=1, seed=41)
    sim.write_across_shots(km.q(0), 1)
    sim.write_across_shots(km.b(0), int(first))
    sim.write_across_shots(km.b(1), int(outer))
    sim.do(km.Circuit(text))
    assert Fraction(sim.read_shot_phase(0)) == (
        angle.half_turns() if first and outer else 0
    )
    amplitudes = torch.tensor([1 + 2j, 3 - 1j, -2 + 5j, 7 + 4j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    expected = amplitudes.clone()
    theta = pi * angle.signed_half_turns_float()
    if first and outer:
        expected[2:] *= torch.exp(torch.tensor(1j * theta, dtype=torch.complex128))
    setup = (
        ClassicalXor((), 3, constant=first).instruction(),
        ClassicalXor((), 4, constant=outer).instruction(),
    )
    gates = binding.bind_reversible_phase_execution(program, classical_offset=3)
    circuit = DynamicCircuit(2, inputs=amplitudes, dtype=torch.complex128).append(
        FieldOperation(2, (*setup, *gates))
    )
    actual = run_dynamic(
        circuit, shots=1, seed=41, strategy="batched"
    ).final_states.reshape(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert all(wire < 2 for gate in gates for wire in gate.wires)


def test_exact_codec_keeps_zero_and_tiny_wrapped_angles() -> None:
    program = Program(
        1, 0, (Op("z_pow", (0,), angle=Angle()), Op("z_pow", (0,), angle=-Angle(1, 0)))
    )
    assert decode_kmx(encode_kmx(program)) == program
    assert "Z_POW q0 0.0\n" in encode_kmx(program)
    # Existing streams keep their byte encoding.
    old = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=0"\nX q0\n'
    assert encode_kmx(decode_kmx(old)) == old


@pytest.mark.parametrize(
    "line",
    ["Z_POW q0", "Z_POW q0 0.5 q1", "Z_POW q0 1e-2", "Z_POW q0 nan", "Z_POW b0 0.5"],
)
def test_malformed_phase_codec_is_rejected(line: str) -> None:
    with pytest.raises(ValueError):
        decode_kmx(
            'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=0"\n' + line
        )


def test_angle_placement_and_undefined_phase_guard_are_rejected() -> None:
    with pytest.raises(ValueError, match="exact PhaseAngle128"):
        Program(1, 0, (Op("z_pow", (0,)),))
    with pytest.raises(ValueError, match="Only Z_POW"):
        Program(1, 0, (Op("x", (0,), angle=Angle()),))
    with pytest.raises(ValueError, match="before definition"):
        Program(1, 1, (Op("z_pow", (0,), condition_bit=0, angle=Angle()),))
    with pytest.raises(ValueError, match="Only HMR"):
        Program(1, 1, (Op("z_pow", (0,), result_bit=0, angle=Angle()),))
    with pytest.raises(CapabilityError, match="cannot represent"):
        lower_reversible_phase(
            CircuitIR(1, (Instruction("phase", (0,), {"theta": 0.1}),))
        )


def test_phase_binding_can_be_replaced_without_changing_its_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Op("z_pow", (0,), angle=Angle.from_integer_half_turns(1))
    program = Program(2, 0, (operation,))
    inputs = torch.tensor([1 + 2j, -3j, 2 - 1j, 4j], dtype=torch.complex128)
    inputs /= inputs.norm()

    def consume() -> torch.Tensor:
        gates = binding.bind_reversible_phase_execution(program)
        circuit = DynamicCircuit(2, inputs=inputs, dtype=torch.complex128).append(
            FieldOperation(2, gates)
        )
        return run_dynamic(circuit, shots=1, seed=41, strategy="batched").final_states

    baseline = consume()
    calls = []

    def replacement(op: Op, metadata: dict[str, object]) -> Instruction:
        calls.append(op)
        assert op.angle == Angle.from_integer_half_turns(1)
        return Instruction("z", op.qubits, metadata=metadata)

    monkeypatch.setattr(binding, "_phase_instruction", replacement)
    torch.testing.assert_close(consume(), baseline, atol=1e-12, rtol=0)
    assert calls == [operation]
