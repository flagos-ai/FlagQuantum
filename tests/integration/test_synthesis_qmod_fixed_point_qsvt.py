"""Verify the pinned fixed-point QSVT and projected arithmetic phases."""

import ast
import cmath
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "fixed_point_qsvt_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path, row


def reference(source: str) -> fq.Circuit:
    match = re.search(r"qsvt_expanded___0\((\[.*?\])\s*,", source, re.S)
    assert match is not None
    angles = ast.literal_eval(match.group(1))
    assert len(angles) == 26
    circuit = fq.Circuit(7, dtype=DTYPE)
    for wire in range(7):
        circuit.h(wire)
    for index, angle in enumerate(angles):
        initial = index % 2 == 0
        if initial:
            for wire in range(6):
                circuit.h(wire)
        phases = []
        for word in range(64):
            a, b, c = word & 3, (word >> 2) & 1, word >> 3
            selected = (word == 0) if initial else (a + b == 3 and a - c == -2)
            for marker in range(2):
                phases.append(cmath.exp(1j * angle * ((marker ^ selected) - 0.5)))
        circuit.any(*range(7), matrix=torch.diag(torch.tensor(phases, dtype=DTYPE)))
        if initial:
            for wire in range(6):
                circuit.h(wire)
    circuit.h(6)
    return circuit


def test_original_fixed_point_qsvt_full_state_and_clean_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(128, -1)
    expected = fq.run(reference(path.read_text())).statevector().flatten()
    assert program.report["workspace_qubits"] == 3
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


PROJECT = """
qstruct Data {a:qnum<1>; b:qnum<1>;}
qperm pred(r:Data,m:qbit) {m ^= (r.a == 1) and (r.b == 0);}
qperm forward(m:qbit,r:qbit[2]) {pred(r,m);}
qperm rotate(theta:real,const m:qbit,const r:Data) {
    within {forward(m,r);} apply {RZ(theta,m);}
}
"""


def test_forwarded_projector_keeps_conjunction_and_absolute_phase() -> None:
    source = """
        qfunc main(output r:Data,output m:qbit) {
            allocate(r); allocate(m); hadamard_transform(r); H(m); rotate(0.7,m,r);
        }
    """
    source = PROJECT + source
    program = synth.synthesize(source, source_format="qmod")
    diagonal = torch.tensor(
        [
            cmath.exp(0.7j * ((marker ^ (word == 1)) - 0.5))
            for word in range(4)
            for marker in range(2)
        ],
        dtype=DTYPE,
    )
    circuit = (
        fq.Circuit(3, dtype=DTYPE)
        .h(0)
        .h(1)
        .h(2)
        .any(0, 1, 2, matrix=torch.diag(diagonal))
    )
    torch.testing.assert_close(
        fq.run(program).statevector(), fq.run(circuit).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("call", ["pred(r,r);", "pred(r,m); X(m);"])
def test_projector_refuses_alias_or_nonforwarding_helper(call: str) -> None:
    source = """
        qfunc main(output r:Data,output m:qbit) {
            allocate(r); allocate(m); rotate(0.7,m,r);
        }
    """
    source = PROJECT.replace("pred(r,m);", call) + source
    with pytest.raises(InteropConversionError):
        synth.synthesize(source, source_format="qmod")
