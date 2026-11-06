"""Compare dense independent Pauli exponentials with Qmod evolution."""

import cmath
import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/suzuki_trotter.qmod"
)
DTYPE = torch.complex128


def _reference(time: float, order: int, repetitions: int) -> fq.Circuit:
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    # The last array index is the most significant bit in the statevector.
    first = (1.5, torch.kron(torch.kron(z, x), x))
    second = (0.5, torch.kron(torch.kron(z, x), y))
    terms = (first, second) if order == 1 else (first, second, second, first)
    step = time / repetitions / (2 if order == 2 else 1)
    circuit = fq.Circuit(3, dtype=DTYPE)
    identity = torch.eye(8, dtype=DTYPE)
    for _ in range(repetitions):
        for coefficient, pauli in terms:
            angle = step * coefficient
            matrix = math.cos(angle) * identity - 1j * math.sin(angle) * pauli
            circuit.any(0, 1, 2, matrix=matrix)
    return circuit


@pytest.mark.parametrize(("order", "repetitions"), [(1, 1), (2, 3)])
def test_suzuki_trotter_matches_independent_circuit(order, repetitions):
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "82b981254ad43471828386aeb8f4ed1546bc6a90"
    time = 0.37
    if order == 1:
        program = _synthesize(SOURCE, classical_inputs={"x": time})
    else:
        source = raw.decode().replace("x, 1, 1", f"x, {order}, {repetitions}")
        program = _synthesize(
            source, source_format="qmod", classical_inputs={"x": time}
        )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(_reference(time, order, repetitions)).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("Pauli::Y", "Pauli::A", "expected Pauli"),
        ("Pauli::X,", "", "word length"),
        ("x, 1, 1", "x, 3, 1", "use order"),
        ("x, 1, 1", "x, 1, 257", "repetitions"),
    ],
)
def test_suzuki_trotter_rejects_invalid_inputs(old, new, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        _synthesize(
            SOURCE.read_text().replace(old, new, 1),
            source_format="qmod",
            classical_inputs={"x": 0.37},
        )


def test_identity_term_preserves_signed_global_phase():
    source = """qfunc main(output q: qbit[]) {
        allocate(1, q);
        suzuki_trotter([PauliTerm {coefficient=-0.5, pauli=[Pauli::I]}],
                       0.6, 1, 1, q);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    expected = torch.tensor([cmath.exp(0.3j), 0], dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )
