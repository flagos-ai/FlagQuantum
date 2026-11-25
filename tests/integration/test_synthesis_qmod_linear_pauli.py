"""Verify Qmod linear Y rotations against an independent block matrix."""

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
    / "fixtures/classiq_qmod/linear_pauli_rotations.qmod"
)
DTYPE = torch.complex128


def _rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)


@pytest.mark.parametrize("superposition", [False, True])
def test_linear_pauli_original_and_all_control_values(superposition: bool) -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "62dab91a568855021a78930b2bb50011121bdf90"
    )
    reference = fq.Circuit(9, dtype=DTYPE)
    if superposition:
        source = raw.decode().replace(
            "linear_pauli_rotations(",
            "hadamard_transform(x); Z(x[0]); linear_pauli_rotations(",
        )
        program = synth.synthesize(source, source_format="qmod")
        for wire in range(6):
            reference.h(wire)
        reference.z(5)
    else:
        program = synth.synthesize(SOURCE)
    blocks = []
    for x in range(64):
        # Qmod's target index zero is the least significant target wire.
        block = torch.kron(_rotation(7.0 * x + 0.33), _rotation(x + 0.3))
        blocks.append(torch.kron(block, _rotation(2.1 * x + 0.1)))
    reference.any(*range(9), matrix=torch.block_diag(*blocks))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[2, 2, 2]", "[2, 3, 2]", "supports only Y"),
        ("[2.1, 1, 7.0]", "[2.1, 1]", "match the target width"),
        ("x, ind);", "x, x);", "disjoint qubit arrays"),
    ],
)
def test_linear_pauli_rejects_unsupported_inputs(old, new, message):
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(SOURCE.read_text().replace(old, new, 1), source_format="qmod")
