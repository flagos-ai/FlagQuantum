"""Structural parity with the locked source Circuit reader and counters."""

import json
import math
from base64 import b64decode
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import (
    concatenate_kmx,
    decode_kmx_or_kmb,
    inspect_kmx,
    repeat_kmx,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_circuit_structure.json").read_text()
)
_KMB_REFUSALS = json.loads(
    (
        Path(__file__).parent / "fixtures" / "kickmix_circuit_kmb_refusals.json"
    ).read_text()
)


@pytest.mark.parametrize("name", tuple(_FIXTURES))
def test_source_file_detection_and_structure(name: str) -> None:
    item = _FIXTURES[name]
    text: str = item["text"]
    expected: str = item["canonical"]
    assert decode_kmx_or_kmb(text.encode()) == expected
    kmb = encode_kmb(text)
    assert decode_kmx_or_kmb(kmb) == expected
    stats = inspect_kmx(text)
    assert inspect_kmx(expected) == stats
    assert [
        stats.qubits,
        stats.bits,
        stats.registers,
        stats.records,
        *stats.quantum_columns,
        stats.classical_targets,
        stats.conditions,
        stats.rotations,
        stats.tags,
        stats.tag_bytes,
        stats.max_condition_depth,
        stats.ccz_state_weight,
    ] == item["stats"]
    assert sum(dict(stats.opcode_counts).values()) == stats.records


@pytest.mark.parametrize("data", [b"\xd7", b"\xd7X q0", b"\xd7\x50\xc7"])
def test_reserved_binary_prefix_is_not_read_as_text(data: bytes) -> None:
    with pytest.raises(ValueError, match="KMB magic"):
        decode_kmx_or_kmb(data)


def test_full_magic_with_truncated_header_is_binary() -> None:
    with pytest.raises(ValueError, match="Truncated KMB"):
        decode_kmx_or_kmb(encode_kmb("")[:16])


@pytest.mark.parametrize("case", tuple(_KMB_REFUSALS))
def test_original_malformed_kmb_cases(case: str) -> None:
    with pytest.raises(ValueError):
        decode_kmb(b64decode(_KMB_REFUSALS[case]["kmb"]))


def test_lossless_bridge_preserves_unused_declared_slots() -> None:
    program = ReversiblePhaseProgram(4, 3, (), (1,))
    text = encode_kmx(program)
    assert decode_kmx(decode_kmx_or_kmb(encode_kmb(text))) == program
    stats = inspect_kmx(text)
    assert (stats.qubits, stats.bits, stats.tags) == (0, 0, 1)


def test_actual_opcode_counts_and_ccz_state_estimate_are_independent() -> None:
    stats = inspect_kmx("CCZ q0 q1 q2\nCCX q0 q1 q2\nZ_POW q0 0.5")
    assert dict(stats.opcode_counts) == {"CCZ": 1, "CCX": 1, "Z_POW": 1}
    assert stats.ccz_state_weight == 3


def test_file_reader_canonicalizes_hash_inside_metadata() -> None:
    raw = b'INFO_TAG "A#B" # comment\nREGISTER r0 "loop#one" # comment\n'
    assert decode_kmx_or_kmb(raw) == 'REGISTER r0 "loop\\Pone"\nINFO_TAG "A\\PB"'


def test_source_composition_keeps_original_physical_ids() -> None:
    left = (
        'REGISTER r0 "left"\nAPPEND_TO_REGISTER q7 r0\nZ_POW q1 -0.25\nINFO_TAG "left"'
    )
    right = 'REGISTER r0 "right"\nAPPEND_TO_REGISTER b9 r0\nCSWAP q1 q2 q3 if b4\nINFO_TAG "right"'
    left_register = 'APPEND_TO_REGISTER q7 r0\nREGISTER r0 "left"'
    left_operations = 'Z_POW q1 1.75\nINFO_TAG "left"'
    right_operations = 'CSWAP q1 q2 q3 if b4\nINFO_TAG "right"'
    expected = "\n".join((left_register, left_operations, right_operations))
    assert concatenate_kmx(left, right) == expected
    assert concatenate_kmx("", right) == (
        'APPEND_TO_REGISTER b9 r0\nREGISTER r0 "right"\n' + right_operations
    )
    assert repeat_kmx(left, 0) == left_register
    assert repeat_kmx(left, 3) == "\n".join((left_register, *([left_operations] * 3)))
    assert repeat_kmx('REGISTER r0 "name"', (1 << 64) - 1) == 'REGISTER r0 "name"'
    with pytest.raises(ValueError, match="record limit"):
        repeat_kmx(left, 2_000_000_000)
    with pytest.raises(ValueError, match="unsigned"):
        repeat_kmx(left, -1)


def test_repeated_phase_circuit_appends_with_expected_complex_amplitudes() -> None:
    source = repeat_kmx("CX q0 q1\nZ_POW q1 0.25", 2)
    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=0"'
    program = decode_kmx(f"{header}\n{decode_kmx_or_kmb(encode_kmb(source))}")
    gates = bind_reversible_phase_execution(program)
    assert [op.opcode for op in program.operations] == ["cx", "z_pow"] * 2
    amplitudes = torch.tensor([1 + 2j, 3 - 1j, -1 + 5j, 2 + 4j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    circuit = DynamicCircuit(2, inputs=amplitudes, dtype=torch.complex128)
    circuit.append(FieldOperation(2, gates))
    result = run_dynamic(circuit, shots=2, seed=11, strategy="batched")
    factors = torch.exp(
        torch.tensor([0, 2, 1, 1], dtype=torch.float64) * (1j * math.pi / 4)
    )
    torch.testing.assert_close(
        result.final_states.reshape(2, -1),
        (amplitudes * factors).expand(2, -1),
        atol=1e-12,
        rtol=0,
    )
