"""Verify the original modular-power QPE and exact phase register order."""

import hashlib
import json
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
    manifest = json.loads((FIXTURES / "shor_sources.json").read_text())
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


def reference() -> tuple[fq.Circuit, torch.Tensor]:
    phase_size, residue_size = 1 << 10, 1 << 5
    circuit = fq.Circuit(15, dtype=DTYPE).x(14)
    eye = torch.eye(residue_size, dtype=DTYPE)
    for bit in range(10):
        wire = 9 - bit
        multiplier = pow(11, 1 << bit, 21)
        perm = torch.zeros_like(eye)
        for value in range(residue_size):
            perm[multiplier * value % 21 if value < 21 else value, value] = 1
        circuit.h(wire).any(wire, *range(10, 15), matrix=torch.block_diag(eye, perm))
    inverse_qft = torch.fft.fft(torch.eye(phase_size, dtype=DTYPE), dim=0, norm="ortho")
    circuit.any(*range(10), matrix=inverse_qft)
    before_fourier = torch.zeros((phase_size, residue_size), dtype=DTYPE)
    for exponent in range(phase_size):
        before_fourier[exponent, pow(11, exponent, 21)] = phase_size**-0.5
    expected = torch.fft.fft(before_fourier, dim=0, norm="ortho")
    return circuit, expected


def test_original_shor_full_state_and_phase_peaks() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    assert program.report["qmod_fractional_bits"] == {"phase_var": 10}
    result = fq.run(program)
    actual = result.statevector().reshape(1 << 15, -1)
    circuit, analytic = reference()
    expected = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(expected, analytic.flatten(), atol=1e-12, rtol=0)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    marginal = analytic.abs().square().sum(dim=1)
    reported = dict(program.probabilities(result, registers=("phase_var",)))
    assert (
        max(
            abs(reported.get((word,), 0.0) - float(probability))
            for word, probability in enumerate(marginal)
        )
        < 1e-11
    )
    assert set(torch.topk(marginal, 6).indices.tolist()) == {0, 171, 341, 512, 683, 853}


@pytest.mark.parametrize(
    ("call", "diagnostic"),
    [
        ("touch(p); period(x,p);", "initialized by allocate and unused"),
        ("period(x,p); H(p);", "terminal output"),
        ("period(x,x);", "unsigned phase bits|disjoint"),
    ],
)
def test_modular_phase_counter_lifetime(call: str, diagnostic: str) -> None:
    source = """
        qfunc touch(p:qbit[]) {H(p[0]);}
        qfunc period(x:qnum,p:qnum) {
            qpe_flexible(lambda(k){modular_multiply_constant_inplace(3,2**k,x);},p);
        }
        qfunc main(output x:qnum<2>,output p:qnum<2,UNSIGNED,2>) {
            allocate(x);allocate(p);x ^= 1;
    """ + call + "}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
