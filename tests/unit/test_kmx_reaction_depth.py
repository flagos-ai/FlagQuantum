"""Source reaction-depth parity on accepted reversible artifact operations."""

import json
import re
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.reaction_depth import (
    reaction_depth,
    touched_qubit_count,
)
from flagquantum.compiler.formats.kmx import decode_kmx
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
