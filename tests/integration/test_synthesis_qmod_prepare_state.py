"""Check original Qmod probability preparation and Bell-state addition."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def _original(name: str, digest: str) -> Path:
    path = FIXTURES / name
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return path


def test_probability_original_against_independent_circuit():
    path = _original(
        "example_prepare_state.qmod", "c49fe290ae660e31f44330eb41f7e65222d47788"
    )
    expected = torch.tensor(
        [0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06], dtype=torch.float64
    )
    identity = torch.eye(8, dtype=DTYPE)
    delta = identity[:, 0] - expected.sqrt()
    matrix = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    reference = fq.Circuit(3, dtype=DTYPE).any(0, 1, 2, matrix=matrix)
    program = synth.synthesize(path)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert (actual.flatten().abs().square() - expected).norm().item() <= 0.01
    assert program.report["workspace_qubits"] == 0


def test_add_bell_states_original_against_independent_circuit():
    path = _original("add_bell_states.qmod", "c5a51feb0f842b76bd2171ed62939bbd1577ffab")
    program = synth.synthesize(path)
    assert tuple(map(len, program.registers.values())) == (2, 2, 3)
    reference = fq.Circuit(7 + program.report["workspace_qubits"], dtype=DTYPE)
    reference.h(0).cx(0, 1).h(2).cx(2, 3).x(3)
    # Independent addition oracle on (a, b, sum), keeping the two Bell states.
    matrix = torch.zeros((128, 128), dtype=DTYPE)
    for index in range(128):
        a = index >> 5
        b = (index >> 3) & 3
        matrix[index ^ (a + b), index] = 1
    reference.any(*range(7), matrix=matrix)

    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    probabilities = actual.reshape(4, 4, 8, -1).abs().square()
    expected = torch.tensor([0.5, 0, 0, 0.5], dtype=torch.float64)
    assert (probabilities.sum(dim=(1, 2, 3)) - expected).norm().item() <= 1e-12
    expected = torch.tensor([0, 0.5, 0.5, 0], dtype=torch.float64)
    assert (probabilities.sum(dim=(0, 2, 3)) - expected).norm().item() <= 1e-12


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("prepare_state([1/0, 0], 0, x);", "denominator must be positive"),
        ("prepare_state([1/y, 0], 0, x);", "integer denominator"),
        ("prepare_state([1/4, 1/4], 0, x);", "sum to one"),
        ("prepare_state([-0.5, 1.5], 0, x);", "nonnegative"),
        ("prepare_state([1, 0], -0.1, x);", "bound must be nonnegative"),
        ("allocate(x); prepare_state([1, 0], 0, x);", "already initialized"),
        ("x = 1/2;", "numeric form is not admitted"),
    ],
)
def test_prepare_state_rejects_invalid_literals_and_reinitialization(body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output x:qbit) {{ {body} }}", source_format="qmod"
        )
