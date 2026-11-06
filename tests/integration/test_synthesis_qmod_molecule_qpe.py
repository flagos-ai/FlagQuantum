"""Independent dense Circuit for the pinned molecular phase estimation."""

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError


def molecule_qpe_reference(source: str) -> fq.Circuit:
    dtype = torch.complex128
    axes = {
        "I": torch.eye(2, dtype=dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    records = re.findall(
        r"SparsePauliTerm\s*\{\s*paulis=\[(.*?)\],\s*coefficient=([-\d.]+)",
        source,
        re.S,
    )
    assert len(records) == 49
    paulis = []
    for record, weight in records:
        word = ["I"] * 6
        for axis, index in re.findall(r"pauli=Pauli::([IXYZ]),\s*index=(\d+)", record):
            word[int(index)] = axis
        matrix = torch.ones((1, 1), dtype=dtype)
        for axis in reversed(word):
            matrix = torch.kron(matrix, axes[axis])
        paulis.append((matrix, float(weight)))

    identity = torch.eye(64, dtype=dtype)
    circuit = fq.Circuit(13, dtype=dtype)
    for wire in range(6, 13):
        circuit.h(wire)
    for bit, wire in enumerate(reversed(range(6, 13))):
        power = 1 << bit
        repetitions = math.ceil(0.05 * power**1.5)
        angle = -math.pi * power / repetitions
        step = identity
        for matrix, weight in (*paulis, *reversed(paulis)):
            factor = (
                math.cos(angle * weight) * identity
                - 1j * math.sin(angle * weight) * matrix
            )
            step = factor @ step
        powered = torch.linalg.matrix_power(step, repetitions)
        circuit.any(wire, *range(6), matrix=torch.block_diag(identity, powered))
    indices = torch.arange(128, dtype=torch.float64)
    inverse_fourier = torch.exp(
        -2j * math.pi * torch.outer(indices, indices) / 128
    ) / math.sqrt(128)
    circuit.any(*range(6, 13), matrix=inverse_fourier)
    return circuit


pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def original() -> Path:
    manifest = json.loads((FIXTURES / "molecule_qpe_sources.json").read_text())
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
    return path


def test_original_molecule_qpe_preserves_the_full_signed_phase_state():
    path = original()
    program = synth.synthesize(path, max_workspace=0)
    actual = fq.run(program).statevector()
    expected = fq.run(molecule_qpe_reference(path.read_text())).statevector()
    assert dict(program.registers) == {
        "state": tuple(range(6)),
        "phase": tuple(range(6, 13)),
    }
    assert program.report["workspace_qubits"] == 0
    assert program.report["qmod_fractional_bits"] == {"phase": 7}
    assert program.report["qmod_signed_registers"] == ("phase",)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    ("repetitions", "diagnostic"),
    [
        ("ceiling(0)", "1..256 repetitions"),
        ("ceiling(1.0/0.0)", "division by zero"),
        ("ceiling(2 ** 9.5)", "bounded real domain"),
        ("ceiling(257.0)", "1..256 repetitions"),
    ],
)
def test_static_repetition_bounds_and_domain(repetitions, diagnostic):
    source = """
    qfunc main(output q:qbit[1]) {
      allocate(q);
      suzuki_trotter(SparsePauliOp {
        terms=[SparsePauliTerm {
          paulis=[IndexedPauli {pauli=Pauli::Y,index=0}], coefficient=0.4
        }],
        num_qubits=1
      }, 0.2, 2, REPETITIONS, q);
    }
    """
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            source.replace("REPETITIONS", repetitions), source_format="qmod"
        )


def test_signed_bit_view_cannot_escape_as_a_helper_output():
    source = """
    qfunc replace_bit(dummy:int, output x:qbit[1]) {allocate(x);}
    qfunc main(output x:qnum<1, True, 1>) {
      replace_bit(0, x);
    }
    """
    with pytest.raises(InteropConversionError, match="borrowed bit view"):
        synth.synthesize(source, source_format="qmod")
