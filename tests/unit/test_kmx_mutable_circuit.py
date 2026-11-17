"""Verify locked mutable circuit parsing, ownership and executable reversal."""

import json
from copy import copy
from io import BytesIO
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmb_operations import OPERATION_CODES
from flagquantum.compiler.formats.kmx import decode_kmx
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

_CASE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_mutable_circuit.json").read_text()
)


@pytest.mark.parametrize("kmx,canonical", _CASE["positive"])
def test_original_single_statement_sequence(kmx: str, canonical: str) -> None:
    builder = KmxCircuitBuilder()
    builder.append_kmx_text(kmx)
    assert builder.freeze().to_kmx() == canonical
    assert decode_kmb(encode_kmb(kmx)) == canonical


@pytest.mark.parametrize("kmx", _CASE["syntax_errors"] + _CASE["boundary_error"])
def test_original_syntax_and_identifier_refusals(kmx: str) -> None:
    builder = KmxCircuitBuilder()
    with pytest.raises(ValueError):
        builder.append_kmx_text(kmx)


def test_statement_failure_preserves_valid_prefix() -> None:
    builder = KmxCircuitBuilder()
    with pytest.raises(ValueError):
        builder.append_kmx_text("X q1\nCCX q0 b1 q2\nZ q7")
    assert builder.freeze().to_kmx() == "X q1"
    with pytest.raises(ValueError):
        builder.append_kmx_line("DEBUG_PRINT q3 bad")
    assert builder.freeze().to_kmx() == "X q1\nDEBUG_PRINT q3"


@pytest.mark.parametrize("kmx", _CASE["aliases"])
def test_original_collision_is_deferred_to_freeze(kmx: str) -> None:
    builder = KmxCircuitBuilder()
    builder.append_kmx_text(kmx)
    assert builder.qubits >= 1
    with pytest.raises(ValueError, match="repeats a quantum wire"):
        builder.freeze()


def test_sparse_boundary_and_missing_columns_at_freeze() -> None:
    builder = KmxCircuitBuilder()
    for kmx in _CASE["boundary_ok"]:
        builder.append_kmx_text(kmx)
    assert builder.qubits == builder.freeze().qubits == 500_000_000
    assert builder.freeze().bits == 500_000_000
    for name in ("X", "CX", "CCX", "BIT_STORE0", "BIT_STORE0_IF", "INFO_TAG"):
        incomplete = KmxCircuitBuilder()
        incomplete._columns.operations.append(OPERATION_CODES[name])
        with pytest.raises(ValueError, match="underflow"):
            incomplete.freeze()
    complete = KmxCircuitBuilder()
    for kmx in (
        "X q1",
        "CX q1 q2",
        "CCX q1 q2 q3",
        "BIT_STORE0 b1",
        "BIT_STORE0 b1 if b1",
        'INFO_TAG "ab"',
        'INFO_TAG ""',
    ):
        complete.append_kmx_line(kmx)
        assert complete.freeze().records[-1].to_kmx() == kmx


def test_original_file_and_bulk_parsing() -> None:
    builder = KmxCircuitBuilder()
    file, expected = _CASE["file"]
    with BytesIO(b"prefix" + file.encode()) as stream:
        stream.seek(6)
        builder.append_kmx_file(stream)
        assert not stream.closed and stream.read() == b""
    assert builder.freeze().to_kmx() == expected
    builder.clear()
    bulk, expected = _CASE["bulk"]
    builder.append_kmx_text(bulk)
    assert builder.freeze().to_kmx() == expected
    assert len(builder.freeze().records) == 37
    assert len(builder.freeze().registers) == 2
    builder.append_kmx_line('INFO_TAG "literal#hash" # comment')
    assert builder.freeze().records[-1].tag == "literal#hash"


def test_owned_append_reversal_and_reset() -> None:
    first = KmxCircuitBuilder()
    first.append_kmx_text('APPEND_TO_REGISTER q4 r0\nINFO_TAG "base"\nX q0')
    other = KmxCircuitBuilder()
    other.append_kmx_text(
        'APPEND_TO_REGISTER b9 r0\nINFO_TAG ""\nZ_POW q1 0.25\nCSWAP q1 q2 q3 if b4'
    )
    other.append_info_tag("last # β")
    frozen = other.freeze()
    first.append(frozen)
    first.append(other)
    first.append_reversed(other)
    assert first.freeze().to_kmx() == _CASE["assembled"]
    assert first.qubits == 5 and first.freeze().bits == 5
    duplicate = copy(first)
    duplicate.append_kmx_line('REGISTER r0 "other"')
    assert first.freeze().registers[0].name == ""
    assert duplicate.freeze().registers[0].name == "other"
    snapshot = first.freeze()
    first.clear()
    assert first.freeze() == read_kmx_circuit("")
    assert snapshot.to_kmx() == _CASE["assembled"]
    reverse = KmxCircuitBuilder()
    reverse.replace_registers(frozen.registers)
    reverse.append_reversed(other)
    restored = KmxCircuitBuilder()
    restored.append_reversed(reverse)
    restored.replace_registers(frozen.registers)
    assert restored.freeze() == frozen


def test_reverse_gate_sequence_matches_complex_execution_and_clean_work() -> None:
    initial = KmxCircuitBuilder()
    initial.append_kmx_text("X q0\nZ_POW q0 0.25\nCX q0 q1")
    actual = copy(initial)
    actual.append_reversed(initial)
    kmx = actual.freeze().to_kmx()
    assert kmx == "X q0\nZ_POW q0 0.25\nCX q0 q1\nCX q0 q1\nZ_POW q0 0.25\nX q0"
    executable = decode_kmx(
        f'INFO_TAG "@flagquantum:reversible-phase-v1 wires=3 bits=0"\n{kmx}'
    )
    state = torch.tensor(
        [1 + 2j, 0, 3 - 1j, 0, -1 + 5j, 0, 2 + 4j, 0], dtype=torch.complex128
    )
    state /= state.norm()
    expected = state.clone()
    expected[:4] *= 1j
    circuit = DynamicCircuit(3, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(3, bind_reversible_phase_execution(executable)))
    final = run_dynamic(
        circuit, shots=2, seed=23, strategy="batched"
    ).final_states.reshape(2, -1)
    torch.testing.assert_close(final, expected.expand(2, -1), atol=1e-12, rtol=0)
    assert not torch.count_nonzero(final[:, 1::2])
