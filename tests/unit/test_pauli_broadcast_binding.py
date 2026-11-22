"""Complete X/Z binding streams and exact source phase consumers."""

import json
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_binding import (
    broadcast_x_ir,
    broadcast_z_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    _port,
    verify_gate_stream,
)

pytestmark = pytest.mark.unit
_Opcode = Literal["x", "z"]
_Emit = Callable[[object], tuple[Instruction, ...]]
_FIXTURE = Path(__file__).with_name("fixtures").joinpath("pauli_broadcast_binding.json")


def _verify_unary_stream(emit: _Emit, port: Port, expected: tuple[int, str]) -> None:
    verify_gate_stream(lambda ports, _: emit(ports[0]), (port,), expected)


def test_complete_scalar_and_broadcast_streams() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    cases: list[tuple[_Opcode, Port, tuple[int, str] | None]] = fixture["cases"]
    assert len(cases) > 100
    for opcode, port, expected in cases:
        emit = broadcast_x_ir if opcode == "x" else broadcast_z_ir
        if expected is None:
            with pytest.raises(ValueError):
                emit(_port(port))
        else:
            _verify_unary_stream(emit, port, expected)


@pytest.mark.parametrize("first,second", list(product((False, True), repeat=2)))
def test_ordered_classical_writes_phase_and_complex_append(
    first: bool, second: bool
) -> None:
    labels = torch.arange(16)
    amplitudes = torch.complex((labels + 1).double(), (23 - labels).double())
    amplitudes /= amplitudes.norm()
    setup = (
        ClassicalXor((), 0, first).instruction(),
        ClassicalXor((), 1, second).instruction(),
    )
    # The second phase reads the newly inverted b0. Repeated literal signs
    # retain their physical records even when their joint phase cancels.
    gates = (
        *broadcast_z_ir((Bit(0), 0, True, True)),
        *broadcast_x_ir((Bit(0), 1, "|->")),
        *broadcast_z_ir((Bit(0), 2)),
        *broadcast_x_ir(("|+>",)),
    )
    expected = amplitudes[labels ^ 4].clone()
    phase = ((labels >> 3) & 1) ^ ((labels >> 1) & 1) ^ int(first) ^ int(not first) ^ 1
    expected *= 1 - 2 * phase
    circuit = DynamicCircuit(4, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(4, (*setup, *gates)))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1)
    )
    assert torch.all(result.classical_bits[:, 0] == (not first))
    assert torch.all(result.classical_bits[:, 1] == second)


def test_repeated_signs_empty_words_and_atomic_rejection() -> None:
    from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
    from flagquantum.core.ir import CircuitIR

    assert broadcast_x_ir(()) == broadcast_z_ir(()) == ()
    for emit, invalid in ((broadcast_x_ir, (0, True)), (broadcast_z_ir, (0, "|->"))):
        with pytest.raises(ValueError):
            emit(invalid)
    program = lower_reversible_phase(CircuitIR(1, broadcast_z_ir((True, True))))
    assert encode_kmx(program).splitlines()[1:] == ["NEG", "NEG"]


def test_complete_original_exact_phase_consumers() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    simple = decode_kmx(fixture["zpow_simple"])
    angles = tuple(op.angle for op in simple.operations)
    assert angles == (
        PhaseAngle128.from_half_turns_exact_double(0.25),
        PhaseAngle128.from_rounded_decimal_half_turns("0.1"),
        PhaseAngle128.from_half_turns_exact_double(0.1),
    )
    assert angles[1] != angles[2]
    assert decode_kmx(encode_kmx(simple)) == simple
    ccz = decode_kmx(fixture["zpow_ccz"])
    assert len(ccz.operations) == 13
    labels = torch.arange(8)
    amplitudes = torch.complex((labels + 1).double(), (2 * labels - 7).double())
    amplitudes /= amplitudes.norm()
    expected = amplitudes.clone()
    expected[-1] *= -1
    circuit = DynamicCircuit(3, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(3, bind_reversible_phase_execution(ccz)))
    actual = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        actual.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=2e-14
    )
