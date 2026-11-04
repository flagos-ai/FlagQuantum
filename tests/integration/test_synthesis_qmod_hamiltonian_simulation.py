"""Check the pinned Classiq guide against an independent Circuit."""

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
    / "fixtures/classiq_qmod/hamiltonian_simulation_guide.qmod"
)
DTYPE = torch.complex128


def _reference(time: float) -> fq.Circuit:
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    identity = torch.eye(2, dtype=DTYPE)
    # Qmod array index 0 is the low bit of the output register.
    terms = (
        (0.7, torch.kron(identity, x)),
        (0.3, torch.kron(z, z)),
        (0.2, torch.kron(x, identity)),
    )
    circuit = fq.Circuit(2, dtype=DTYPE)
    for coefficient, pauli in terms:
        angle = time * coefficient
        unitary = (
            math.cos(angle) * torch.eye(4, dtype=DTYPE) - 1j * math.sin(angle) * pauli
        )
        circuit.any(0, 1, matrix=unitary)
    return circuit


@pytest.mark.parametrize("time", [0.37, -0.81])
def test_original_hamiltonian_simulation_matches_full_statevector(time):
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "523a87121cc5f69215d6bf1a99d19d2c9bc9f331"
    program = synth.synthesize(SOURCE, classical_inputs={"t": time})
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(_reference(time)).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("num_qubits=2", "num_qubits=3", "num_qubits must match"),
        ("index=1", "index=0", "distinct and in bounds"),
        ("index=1", "index=2", "distinct and in bounds"),
        ("index=1", "wrong=1", "IndexedPauli requires distinct fields"),
        ("index=1", "pauli=Pauli::Y", "IndexedPauli requires distinct fields"),
        ("Pauli::X", "Pauli::A", "expected Pauli"),
    ],
)
def test_exponentiate_rejects_malformed_sparse_operators(old, new, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            SOURCE.read_text().replace(old, new, 1),
            source_format="qmod",
            classical_inputs={"t": 0.37},
        )


def test_sparse_identity_preserves_global_phase_and_signed_values():
    source = """qfunc main(output q: qbit[]) {
        allocate(1, q);
        hadamard_transform(q);
        exponentiate(SparsePauliOp {
            num_qubits=1,
            terms=[SparsePauliTerm {coefficient=-0.5, paulis=[]}]
        }, -0.6, q);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    expected = torch.full((2,), cmath.exp(-0.3j) / math.sqrt(2), dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


def test_inline_sparse_operator_also_works_with_suzuki_trotter():
    source = SOURCE.read_text().replace("exponentiate(", "suzuki_trotter(")
    source = source.replace("}, t, qba);", "}, t, 1, 1, qba);")
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"t": 0.37}
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(_reference(0.37)).statevector(),
        atol=1e-12,
        rtol=0,
    )
