"""Validate state preparation through Qmod top-level real constants."""

import hashlib
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/prepare_state.qmod"
)


def test_original_named_probabilities_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "4a84e2058fa7c5b2b90be7ae3019853204b601f0"
    )
    match = re.search(r"probabilities: real\[\] = \[(.*?)\]", raw.decode(), re.S)
    assert match is not None
    expected = torch.tensor(
        [float(field) for field in match[1].split(",") if field.strip()],
        dtype=torch.float64,
    )
    identity = torch.eye(32, dtype=torch.complex128)
    delta = identity[:, 0] - expected.sqrt()
    matrix = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    reference = fq.Circuit(5, dtype=torch.complex128)
    reference.any(*range(5), matrix=matrix)
    program = synth.synthesize(SOURCE)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert torch.linalg.vector_norm(actual.flatten().abs().square() - expected) <= 0.01
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("declarations", "arguments", "diagnostic"),
    [
        ("bound:real=0;", "missing,bound,x", r"declared real\[\] vector"),
        ("bound:real=0;", "bound,bound,x", r"declared real\[\] vector"),
        ("p:real[]=[0.5,0.5];", "p,p,x", "scalar real"),
        ("p:real[]=[0.5,0.5];", "p,missing,x", "real decimal literal"),
        ("p:real[]=[0.5,0.5]; p:real=0;", "p,0,x", "duplicate or reserved"),
    ],
)
def test_invalid_preparation_constants(declarations, arguments, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"{declarations} qfunc main(output x:qbit[]) {{prepare_state({arguments});}}",
            source_format="qmod",
        )


def test_named_amplitude_array_preserves_signs():
    program = synth.synthesize(
        """vector:real[] = [0.6, -0.8]; bound:real=0;
        qfunc main(output x:qbit[]) {prepare_amplitudes(vector,bound,x);}""",
        source_format="qmod",
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        torch.tensor([0.6, -0.8], dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
