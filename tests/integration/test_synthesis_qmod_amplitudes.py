"""Check pinned real amplitude preparation against a Householder Circuit."""

import hashlib
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    ("name", "blob_sha", "width", "bound"),
    [
        (
            "example_prepare_amplitudes",
            "fa7563598bbe60e6ea37a6f5dc888b11593b8744",
            3,
            0.0,
        ),
        (
            "approximated_state_preparation",
            "779137a6172a0b76c3346457d23660a75e0df6b1",
            6,
            0.3,
        ),
        (
            "auxiliary_management",
            "0afcedfa7b70ae3fe804520e0a4735fc94834b21",
            4,
            0.2,
        ),
    ],
)
def test_original_amplitudes_against_independent_householder(
    name, blob_sha, width, bound
):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob_sha
    match = re.search(r"prepare_amplitudes\(\s*\[(.*?)\]", raw.decode(), re.S)
    assert match is not None
    values = [float(field.strip().strip("()")) for field in match[1].split(",")]
    amplitudes = torch.tensor(values, dtype=torch.complex128)
    assert amplitudes.numel() == 1 << width

    identity = torch.eye(1 << width, dtype=torch.complex128)
    delta = identity[:, 0] - amplitudes / amplitudes.norm()
    unitary = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    reference = fq.Circuit(width, dtype=torch.complex128).any(
        *range(width), matrix=unitary
    )
    program = synth.synthesize(source)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert (
        torch.linalg.vector_norm(actual.flatten() - amplitudes).item() <= bound + 1e-12
    )
    assert program.report["workspace_qubits"] == 0


def test_prepare_amplitudes_preserves_negative_global_sign():
    program = synth.synthesize(
        "qfunc main(output x:qbit) { prepare_amplitudes([-1, 0], 0, x); }",
        source_format="qmod",
    )
    actual = fq.run(program).statevector().flatten()
    expected = torch.tensor([-1, 0], dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("output", "body", "diagnostic"),
    [
        ("qnum", "prepare_amplitudes([1, 0, 0], 0, x);", "power of two"),
        ("qnum", "prepare_amplitudes([0, 0], 0, x);", "normalized"),
        ("qnum", "prepare_amplitudes([1, 1], 0, x);", "normalized"),
        ("qnum", "prepare_amplitudes([1, 0], -0.1, x);", "nonnegative"),
        ("qnum<2>", "prepare_amplitudes([1, 0], 0, x);", "match the output width"),
        (
            "qbit",
            "allocate(x); prepare_amplitudes([1, 0], 0, x);",
            "already initialized",
        ),
    ],
)
def test_invalid_amplitude_preparation(output, body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output x:{output}) {{ {body} }}", source_format="qmod"
        )
