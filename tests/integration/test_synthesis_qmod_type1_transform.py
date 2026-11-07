"""Validate the pinned Type I transform via a Fourier parity decomposition."""

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
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/qct_qst_type1.qmod"
)
DTYPE = torch.complex128


def _reference_matrix(width: int) -> torch.Tensor:
    size = 1 << width
    indices = torch.arange(size, dtype=torch.float64)
    fourier = torch.exp(
        2j * math.pi / size * torch.outer(indices, indices)
    ) / math.sqrt(size)
    basis = torch.eye(size, dtype=DTYPE)
    half = size // 2
    even = [(basis[j] + basis[-j]) / math.sqrt(2) for j in range(1, half)]
    odd = [(basis[j] - basis[-j]) / math.sqrt(2) for j in range(1, half)]
    parity = torch.stack([basis[0], *even, basis[half], *odd], dim=1)
    return parity.mH @ fourier @ parity


def test_original_type1_transform_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "ff91c98c34f405b7ebaeb40259a1bc61b58c5460"
    )
    match = re.search(r"prepare_amplitudes\(\s*\[(.*?)\]", raw.decode(), re.S)
    assert match is not None
    amplitudes = torch.tensor(
        [float(field) for field in match[1].split(",")], dtype=DTYPE
    )
    amplitudes /= amplitudes.norm()
    basis = torch.eye(16, dtype=DTYPE)
    delta = basis[:, 0] - amplitudes
    prepare = basis - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    reference = fq.Circuit(4, dtype=DTYPE)
    reference.any(*range(4), matrix=prepare)
    reference.any(*range(4), matrix=_reference_matrix(4))
    program = synth.synthesize(SOURCE)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(("width", "index"), [(1, 1), *((4, i) for i in range(16))])
def test_type1_basis_columns(width: int, index: int) -> None:
    program = synth.synthesize(
        f"""qfunc main(output x:qnum<{width}>) {{
            x={index}; qct_qst_type1(x);
        }}""",
        source_format="qmod",
    )
    expected = _reference_matrix(width)[:, index]
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("body", "width", "diagnostic"),
    [
        ("qct_qst_type1(x);", 4, "initialized registers"),
        ("allocate(x); qct_qst_type1(x);", 5, "bounded Type I"),
        ("allocate(x); qct_qst_type1(x,x);", 4, "one data register"),
    ],
)
def test_type1_refuses_invalid_calls(body: str, width: int, diagnostic: str) -> None:
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output x:qnum<{width}>) {{{body}}}", source_format="qmod"
        )
