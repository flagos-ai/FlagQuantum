"""Source reaction-depth parity on accepted reversible artifact operations."""

import json
import re
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.reaction_depth import (
    reaction_depth,
    reaction_depth_for_operations,
    touched_qubit_count,
)
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseOperation
from flagquantum.compiler.formats.kmx import decode_kmx
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.formats.kmx_debug import decode_kmx_debug

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_reaction_depth.json").read_text()
)


@pytest.mark.parametrize("index", range(len(_CASES)))
def test_locked_reaction_depth(index: int) -> None:
    case = _CASES[index]
    body = "\n".join(case["ops"])
    qubits = 1 + max(map(int, re.findall(r"\bq([0-9]+)", body)), default=0)
    bits = 1 + max(map(int, re.findall(r"\bb([0-9]+)", body)), default=-1)
    inputs = f" inputs={','.join(map(str, range(bits)))}" if bits else ""
    header = f'INFO_TAG "@flagquantum:reversible-phase-v1 wires={qubits} bits={bits}{inputs}"'
    program = decode_kmx(f"{header}\n{body}" if body else header)
    assert reaction_depth(program) == case["reaction_depth"]


def test_touched_qubits_ignore_unused_declared_width() -> None:
    program = decode_kmx(
        'INFO_TAG "@flagquantum:reversible-phase-v1 wires=9 bits=1 inputs=0"\n'
        "CSWAP q0 q1 q2\nCSWAP q3 q4 q5 if b0\n"
    )
    assert touched_qubit_count(program) == 6
    assert reaction_depth(program) == 1


def test_touched_debug_wires_are_counted_without_changing_depth() -> None:
    program, _, _, debug = decode_kmx_debug(
        'INFO_TAG "@flagquantum:reversible-phase-v1 wires=4 bits=1 inputs=0"\n'
        "DEBUG_PRINT q3 if b0\nX q0\nDEBUG_PRINT b0\nDEBUG_PRINT q0\n"
    )
    wires = (
        item.operand[1]
        for item in debug
        if item.operand is not None and item.operand[0] == "q"
    )
    assert touched_qubit_count(program, debug_wires=wires) == 2
    assert reaction_depth(program) == 0


def test_raw_metadata_and_debug_records_preserve_source_reaction_depth() -> None:
    import kickmix as km

    source = (
        'INFO_TAG "@enter:outer"\nCCX q0 q1 q2\n'
        "HMR q2 b0\nDEBUG_PRINT q0 if b0\n"
        'DEBUG_PRINT b0\nDEBUG_PRINT\nINFO_TAG "@exit:outer"'
    )
    circuit = read_kmx_circuit(source)
    operations = (
        ReversiblePhaseOperation(
            record.opcode.lower(),
            record.qubits,
            record.result_bit,
            record.condition_bit,
            record.angle,
        )
        for record in circuit
    )
    assert (
        reaction_depth_for_operations(operations) == km.Circuit(source).reaction_depth()
    )
