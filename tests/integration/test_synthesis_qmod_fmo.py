"""Compare original FMO evolution with independent dense matrix evolution."""

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
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/quantum_walk_fmo.qmod"
)
# Transcribed in source order, with each word ordered most significant wire first.
TERMS = (
    ("III", 226.875),
    ("IIX", -32.0),
    ("IXX", 13.75),
    ("XXX", -19.5),
    ("ZXX", -2.25),
    ("YYX", -20),
    ("IZX", -1.0),
    ("XZX", 12.75),
    ("ZZX", -21.0),
    ("XIX", -15.75),
    ("ZIX", -52.0),
    ("YXY", -15),
    ("IYY", 16.25),
    ("XYY", 15.5),
    ("ZYY", 0.25),
    ("YZY", 16.75),
    ("YIY", -11.75),
    ("IIZ", -11.875),
    ("XXZ", 3.5),
    ("ZXZ", 1.0),
    ("YYZ", 5.5),
    ("IZZ", -33.125),
    ("XZZ", -6.0),
    ("ZZZ", 41.875),
    ("XIZ", 2.5),
    ("ZIZ", -66.875),
    ("IXI", 3.0),
    ("XXI", -6.0),
    ("ZXI", 4.0),
    ("YYI", -4),
    ("IZI", 118.125),
    ("XZI", 0.5),
    ("ZZI", 13.125),
    ("XII", 9.0),
    ("ZII", -8.125),
)
DTYPE = torch.complex128


@pytest.mark.parametrize("time", [0.037, -0.081])
def test_original_fmo_matches_full_state(time: float) -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "f8a538767b019c1cd8600afb30bcc213a110c6cd"
    )
    program = synth.synthesize(SOURCE, classical_inputs={"time_point": time})
    axes = {
        "I": torch.eye(2, dtype=DTYPE),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE),
        "Z": torch.diag(torch.tensor([1, -1], dtype=DTYPE)),
    }
    identity = torch.eye(8, dtype=DTYPE)
    step = identity.clone()
    for word, coefficient in TERMS:
        angle = time * coefficient / 10
        a, b, c = (axes[axis] for axis in word)
        pauli = torch.kron(torch.kron(a, b), c)
        step = (math.cos(angle) * identity - 1j * math.sin(angle) * pauli) @ step
    reference = fq.Circuit(3, dtype=DTYPE)
    reference.any(0, 1, 2, matrix=torch.linalg.matrix_power(step, 10))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_fmo_rejects_nonzero_imaginary_coefficient() -> None:
    with pytest.raises(InteropConversionError, match="expected '0'"):
        synth.synthesize(
            SOURCE.read_text().replace("+ 0j", "+ 1j", 1),
            source_format="qmod",
            classical_inputs={"time_point": 0.037},
        )
