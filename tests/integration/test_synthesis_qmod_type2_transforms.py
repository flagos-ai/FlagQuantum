"""Validate pinned Type II transforms against a matrix Circuit oracle."""

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
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _reference(amplitudes: torch.Tensor, name: str) -> fq.Circuit:
    size = len(amplitudes)
    width = size.bit_length() - 1
    combined = name == "qct_qst_type2"
    basis = torch.eye(size, dtype=torch.complex128)
    delta = basis[:, 0] - amplitudes / amplitudes.norm()
    prepare = basis - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    circuit = fq.Circuit(width + int(combined), dtype=torch.complex128)
    circuit.any(*range(width), matrix=prepare)

    # Read the Fourier coefficients of each half-sample basis vector.
    sample = torch.arange(size, dtype=torch.float64) + 0.5
    modes = torch.arange(size + 1, dtype=torch.float64)
    fourier = torch.exp(1j * math.pi * torch.outer(sample, modes) / size).T
    cosine = fourier[:-1].real.clone()
    sine = fourier[1:].imag.clone()
    cosine[0] *= 2**-0.5
    sine[-1] *= 2**-0.5
    scale = math.sqrt(2 / size)
    if combined:
        circuit.h(width)
        return circuit.any(
            width, *range(width), matrix=scale * torch.block_diag(cosine, -sine)
        )
    return circuit.any(
        *range(width), matrix=scale * (sine if name == "qst_type2" else cosine)
    )


@pytest.mark.parametrize(
    ("name", "digest", "first"),
    [
        ("qct_type2", "12b820aa8fe19a039f18b825b1016892406d7dab", 0.92071884),
        ("qst_type2", "2541d8808bca009faf0ccb62bb985c5db604f240", 0.90482984),
        ("qct_qst_type2", "73f5156921cef9a6ed427ac13c910d0d23312ad8", 0.65104654),
    ],
)
def test_original_type2_transforms_match_matrix_circuit(name, digest, first):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    match = re.search(r"prepare_amplitudes\(\s*\[(.*?)\]", raw.decode(), re.S)
    assert match is not None
    amplitudes = torch.tensor(
        [float(field.strip().strip("()")) for field in match[1].split(",")],
        dtype=torch.complex128,
    )
    program = synth.synthesize(source)
    actual = fq.run(program).statevector()
    expected = fq.run(_reference(amplitudes, name)).statevector()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert actual.flatten()[0].real.item() == pytest.approx(first, abs=1e-8)
    if name == "qct_qst_type2":
        assert actual.flatten()[1].real.item() == pytest.approx(-0.63981132, abs=1e-8)
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("outputs", "body", "diagnostic"),
    [
        ("output x:qnum<2>", "qct_type2(x);", "initialized registers"),
        ("output x:qnum<5>", "allocate(x); qst_type2(x);", "bounded Type II"),
        ("output x:qbit", "allocate(x); qct_qst_type2(x,x);", "disjoint"),
        ("output x:qnum<2>", "allocate(x); qct_qst_type2(x);", "selector"),
    ],
)
def test_type2_refuses_invalid_registers(outputs, body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(f"qfunc main({outputs}) {{ {body} }}", source_format="qmod")
