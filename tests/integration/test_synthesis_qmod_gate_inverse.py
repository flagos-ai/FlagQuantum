"""Verify the original randomized benchmarking gate blocks and their inverse."""

import hashlib
import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/randomized_benchmarking.qmod"
)
DTYPE = torch.complex128


def _forward_matrix(source: str) -> torch.Tensor:
    matrices = {
        "RZ": torch.diag(torch.tensor([-1j, 1j], dtype=DTYPE)),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
        "I": torch.eye(2, dtype=DTYPE),
        "SX": torch.tensor([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=DTYPE) / 2,
    }
    result = matrices["I"]
    forward = source.split("invert", 1)[0]
    for name in re.findall(r"\b(RZ|X|I|SX)\(", forward):
        result = matrices[name] @ result
    return result


def test_original_randomized_benchmarking_round_trip():
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "4b3b3a46668181ac59f525e19421f94bdb57fddb"
    )
    program = synth.synthesize(SOURCE)
    reference = fq.Circuit(1, dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("basis", [0, 1, None])
def test_gate_blocks_against_complex_matrix(inverse, basis):
    source = SOURCE.read_text()
    forward, undo = source.split("invert", 1)
    if inverse:
        source = (
            forward.split("allocate(target);", 1)[0] + "allocate(target); invert" + undo
        )
    else:
        source = forward + "}"
    reference = fq.Circuit(1, dtype=DTYPE)
    if basis is None:
        preparation = "H(target); S(target);"
        reference.h(0).s(0)
    elif basis == 1:
        preparation = "X(target);"
        reference.x(0)
    else:
        preparation = ""
    source = source.replace("allocate(target);", f"allocate(target); {preparation}", 1)
    matrix = _forward_matrix(SOURCE.read_text())
    reference.any(0, matrix=matrix.mH if inverse else matrix)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_noncommuting_inverse_keeps_order_and_phase():
    source = """qfunc main(output target:qbit[1]) {
        allocate(target); H(target); S(target);
        invert {SX(target); RZ(pi,target);}
    }"""
    reference = fq.Circuit(1, dtype=DTYPE).h(0).s(0)
    reference.rz(0, theta=-math.pi).sxdg(0)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_inverse_refuses_allocation():
    with pytest.raises(InteropConversionError, match="static single-qubit"):
        synth.synthesize(
            "qfunc main(output q:qbit) {invert {allocate(q);}}", source_format="qmod"
        )
