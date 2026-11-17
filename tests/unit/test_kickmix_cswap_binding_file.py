"""Verify the locked controlled-swap binding and coherent execution."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reaction_depth import reaction_depth
from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_cswap import broadcast_cswap_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    ternary_matrix,
    verify_gate_stream,
)


@pytest.mark.parametrize("columns,expected", tuple(ternary_matrix("cswap")))
def test_source_cswap_streams(
    columns: tuple[Port, ...], expected: tuple[int, str]
) -> None:
    verify_gate_stream(lambda ports, _: broadcast_cswap_ir(*ports), columns, expected)


@pytest.mark.parametrize("flag", [False, True])
def test_mixed_cswap_append_preserves_full_complex_amplitudes(flag: bool) -> None:
    gates = broadcast_cswap_ir((0, Bit(0), True), (1, 2, 3), (4, 5, 6))
    labels = torch.arange(128)
    basis = labels.to(torch.float64)
    amplitudes = torch.complex(basis + 1, 2 * basis - 7)
    amplitudes /= amplitudes.norm()
    mapped = labels.clone()
    for active, left, right in ((labels >> 6, 5, 2), (int(flag), 4, 1), (1, 3, 0)):
        delta = ((mapped >> left) ^ (mapped >> right)) & active & 1
        mapped ^= (delta << left) | (delta << right)
    expected = torch.empty_like(amplitudes)
    expected[mapped] = amplitudes
    initial = ClassicalXor((), 0, constant=flag).instruction()
    circuit = DynamicCircuit(7, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(7, (initial, *gates)))
    assert encode_kmx(lower_reversible_phase(circuit.to_ir())).splitlines()[
        2 + int(flag) :
    ] == ["CSWAP q0 q1 q4", "SWAP q2 q5 if b0", "SWAP q3 q6"]
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), atol=1e-12, rtol=0
    )
    assert torch.all(result.classical_bits[:, 0] == flag)


def test_cswap_empty_and_invalid_widths_targets_and_aliases() -> None:
    assert broadcast_cswap_ir("|->", (), ()) == ()
    with pytest.raises(ValueError, match="Inconsistent broadcast sizes"):
        broadcast_cswap_ir(0, (1,), (2, 3))
    with pytest.raises(ValueError, match="quantum-only"):
        broadcast_cswap_ir(False, Bit(1), 3)
    with pytest.raises(ValueError):
        broadcast_cswap_ir(0, 0, 1)


@pytest.mark.parametrize("flag", [False, True])
def test_source_guarded_fredkin_kmx_roundtrip_and_complete_amplitude(
    flag: bool,
) -> None:
    text = (
        'INFO_TAG "@flagquantum:reversible-phase-v1 wires=6 bits=1 inputs=0"\n'
        "CSWAP q0 q1 q2\nCSWAP q3 q4 q5 if b0\n"
    )
    header, body = text.split("\n", 1)
    transported = decode_kmb(encode_kmb(body))
    program = decode_kmx(f"{header}\n{transported}\n")
    assert encode_kmx(program) == text
    assert reaction_depth(program) == 1
    assert [op.opcode for op in program.operations] == ["cswap", "cswap"]
    labels = torch.arange(64)
    basis = labels.to(torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 5)
    amplitudes /= amplitudes.norm()
    delta1 = ((labels >> 4) ^ (labels >> 3)) & (labels >> 5) & 1
    delta2 = ((labels >> 1) ^ labels) & (labels >> 2) & int(flag)
    mapped = labels ^ (delta1 << 4) ^ (delta1 << 3) ^ (delta2 << 1) ^ delta2
    expected = torch.empty_like(amplitudes)
    expected[mapped] = amplitudes
    initialization = ClassicalXor((), 0, constant=flag).instruction()
    circuit = DynamicCircuit(6, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(
        FieldOperation(6, (initialization, *bind_reversible_phase_execution(program)))
    )
    result = run_dynamic(circuit, shots=2, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), atol=1e-12, rtol=0
    )
