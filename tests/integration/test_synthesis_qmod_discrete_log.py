"""Verify the original discrete log and symbolic Output allocation."""

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
    manifest = json.loads((FIXTURES / "discrete_log_sources.json").read_text())
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
    circuit = fq.Circuit(14, dtype=DTYPE).x(13)
    eye = torch.eye(16, dtype=DTYPE)
    for offset, base in ((0, 3), (5, 7)):
        for bit in range(5):
            wire = offset + 4 - bit
            multiplier = pow(base, 1 << bit, 13)
            perm = torch.zeros_like(eye)
            for value in range(16):
                perm[multiplier * value % 13 if value < 13 else value, value] = 1
            circuit.h(wire).any(
                wire, *range(10, 14), matrix=torch.block_diag(eye, perm)
            )
    inverse_qft = torch.fft.fft(torch.eye(32, dtype=DTYPE), dim=0, norm="ortho")
    circuit.any(*range(5), matrix=inverse_qft).any(*range(5, 10), matrix=inverse_qft)
    state = torch.zeros((32, 32, 16), dtype=DTYPE)
    for a in range(32):
        for b in range(32):
            state[a, b, (pow(3, a, 13) * pow(7, b, 13)) % 13] = 1 / 32
    return circuit, torch.fft.fftn(state, dim=(0, 1), norm="ortho")


def test_original_discrete_log_full_state_and_two_axis_marginal() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    assert program.report["qmod_fractional_bits"] == {"alpha": 5, "beta": 5}
    result = fq.run(program)
    actual = result.statevector().reshape(1 << 14, -1)
    circuit, analytic = reference()
    expected = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(expected, analytic.flatten(), atol=1e-12, rtol=0)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    marginal = analytic.abs().square().sum(dim=-1)
    reported = dict(program.probabilities(result, registers=("alpha", "beta")))
    assert (
        max(
            abs(reported.get((a, b), 0.0) - float(marginal[a, b]))
            for a in range(32)
            for b in range(32)
        )
        < 1e-11
    )


def test_symbolic_output_shape_survives_nested_helpers() -> None:
    source = """
        qfunc inner(n:int,output x:qnum) {allocate(ceiling(log(n,2))+1,x);x ^= 5;}
        qfunc outer(n:int,output x:qnum) {inner(n,x);}
        qfunc main(output x:qnum) {outer(5,x);}
    """
    program = synth.synthesize(source, source_format="qmod")
    result = fq.run(program)
    assert result.statevector().numel() == 16
    assert dict(program.probabilities(result, registers=("x",))) == pytest.approx(
        {(5,): 1.0}
    )


@pytest.mark.parametrize(
    ("call", "diagnostic"),
    [
        ("make(1,x,y);", "static integer of at least two"),
        ("make(5,x,x);", "writable helper arguments must be disjoint"),
    ],
)
def test_symbolic_output_refuses_invalid_width_or_alias(
    call: str, diagnostic: str
) -> None:
    source = """
        qfunc make(n:int,output a:qnum,output b:qnum) {
            allocate(ceiling(log(n,2))+1,a);allocate(1,b);
        }
        qfunc main(output x:qnum,output y:qnum) {
    """ + call + "}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
