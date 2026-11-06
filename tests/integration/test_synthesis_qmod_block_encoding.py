"""Validate the original checkerboard block encoding and bounded view rules."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original(name: str = "checkerboard_BE_N8.qmod") -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "block_encoding_sources.json").read_text())
    row = next(entry for entry in manifest["fixtures"] if entry["file"] == name)
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


def reference() -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=DTYPE)
    for wire in (4, 5, 6, 1, 2, 3):
        circuit.h(wire)
    circuit.swap(2, 6).swap(1, 5).cx(3, 4)
    # RX(2 acos(a)) has diagonal a and off-diagonal -i sqrt(1-a**2).
    table = []
    for value in (-0.69, -0.34):
        off_diagonal = -1j * math.sqrt(1 - value**2)
        table.append(
            torch.tensor([[value, off_diagonal], [off_diagonal, value]], dtype=DTYPE)
        )
    circuit.any(3, 0, matrix=torch.block_diag(*table))
    for wire in (1, 2, 3):
        circuit.h(wire)
    return circuit


def test_checkerboard_original_full_state_and_all_unitary_columns() -> None:
    path, _ = original()
    program = synth.synthesize(path, max_workspace=0)
    expected = reference()
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )
    inputs = torch.eye(128, dtype=DTYPE)
    actual = fq.Circuit.from_ir(program.circuit, dtype=DTYPE, inputs=inputs, bsz=128)
    independent = fq.Circuit.from_ir(
        expected.to_ir(), dtype=DTYPE, inputs=inputs, bsz=128
    )
    torch.testing.assert_close(actual.state(), independent.state(), atol=1e-12, rtol=0)
    assert dict(program.registers) == {"data": (0,), "s": (1, 2, 3), "j": (4, 5, 6)}
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("bits:qbit[]; q -> bits; H(bits[0]);", "restore every formal port"),
        ("lo:qnum<1>; hi:qnum<1>; q -> {lo,hi}; {hi,lo} -> q;", "original wire order"),
        ("RX(acos(1.01), t);", "acos requires an argument"),
        ("RX(acos(q), t);", "angle must be a literal"),
        ("control(q == 4) {RX(0.4,t);}", "equality value must fit"),
    ],
)
def test_checked_numeric_angle_and_binding_refusals(body: str, diagnostic: str) -> None:
    source = "qfunc layer(q:qnum<2>,t:qbit) {" + body + "}"
    source += "qfunc main(output q:qnum<2>,output t:qbit) {allocate(q);allocate(t);layer(q,t);}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_original_tridiagonal_exact_amplitudes_are_not_normalized() -> None:
    path, _ = original("tridiagonal_BE_size16.qmod")
    assert math.hypot(0.5774, 0.5774, 0.5774) - 1 > 1e-5
    with pytest.raises(
        InteropConversionError, match="inplace preparation requires a normalized"
    ):
        synth.synthesize(path)


@pytest.mark.parametrize("inverse", [False, True])
def test_real_inplace_preparation_and_inverse(inverse: bool) -> None:
    prep = "inplace_prepare_amplitudes([0.6,(-0.8)],0,q);"
    text = "qfunc main(output q:qnum<1>) {allocate(q);" + prep
    if inverse:
        text += "invert {" + prep + "}"
    program = synth.synthesize(text + "}", source_format="qmod")
    state = fq.run(program).statevector().flatten()
    target = [1, 0] if inverse else [0.6, -0.8]
    torch.testing.assert_close(
        state, torch.tensor(target, dtype=DTYPE), atol=1e-12, rtol=0
    )
