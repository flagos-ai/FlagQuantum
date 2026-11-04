"""Check original VarQITE Pauli rotations against independent dense matrices."""

import cmath
import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/variational_quantum_imaginary_time_evolution.qmod"
)
ANGLES = (0.13, -0.29, 0.41, -0.57, 0.68, -0.83)
DTYPE = torch.complex128


def test_original_varqite_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "45875e31fddf68e9632687358ec55da086e45dbb"
    )
    program = synth.synthesize(SOURCE, classical_inputs={"angles": ANGLES})
    reference = fq.Circuit(4, dtype=DTYPE)
    identity = torch.eye(16, dtype=DTYPE)
    axes = {
        "I": torch.eye(2, dtype=DTYPE),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE),
        "Z": torch.diag(torch.tensor([1, -1], dtype=DTYPE)),
    }
    for wire in range(4):
        reference.h(wire)
    for word, angle in zip(
        ("IYZI", "IIZY", "YIZI", "IZIY", "YZII", "YIIZ"), ANGLES, strict=True
    ):
        pauli = torch.ones((1, 1), dtype=DTYPE)
        # Qmod index zero corresponds to the least significant Circuit wire.
        for axis in reversed(word):
            pauli = torch.kron(pauli, axes[axis])
        matrix = math.cos(angle) * identity - 1j * math.sin(angle) * pauli
        reference.any(0, 1, 2, 3, matrix=matrix)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_identity_word_keeps_global_phase() -> None:
    program = synth.synthesize(
        """qfunc main(output q:qbit[]) {
            allocate(1,q); H(q[0]); single_pauli_exponent([Pauli::I],-0.3,q);
        }""",
        source_format="qmod",
    )
    expected = torch.full((2,), cmath.exp(0.3j) / math.sqrt(2), dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("Pauli::Y", "Pauli::A", "expected Pauli"),
        ("Pauli::Y,", "", "word length"),
        ("angles[0]", "angles[6]", "out of bounds"),
    ],
)
def test_single_pauli_rejects_invalid_operands(before, after, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            SOURCE.read_text().replace(before, after, 1),
            source_format="qmod",
            classical_inputs={"angles": ANGLES},
        )
