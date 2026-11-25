"""Independent full-state Circuit for the pinned AQC schedule."""

import ast
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


def aqc_reference(source: str) -> fq.Circuit:
    dtype = torch.complex128
    identity = torch.eye(2, dtype=dtype)
    axes = {
        "I": identity,
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    groups = []
    pattern = r"\[\s*PauliTerm\s*\{.*?\n  \]"
    tables = re.findall(pattern, source, re.S)
    assert len(tables) == 2
    for table in tables:
        terms = re.findall(
            r"PauliTerm\s*\{\s*pauli=\[(.*?)\],\s*coefficient=([-\d.]+)", table, re.S
        )
        group = []
        for word, weight in terms:
            letters = re.findall(r"Pauli::([IXYZ])", word)
            assert len(letters) == 3
            a, b, c = (axes[letter] for letter in reversed(letters))
            group.append((torch.kron(torch.kron(a, b), c), float(weight)))
        groups.append(group)
    schedule = re.findall(
        r"suzuki_trotter\(H([01]), ([-\d.]+), (\d+), (\d+), qba\)", source
    )
    assert len(schedule) == 200
    values = re.search(r"prepare_state\((\[.*?\]), 0", source, re.S)
    assert values
    probabilities = ast.literal_eval(values[1])
    vector = torch.tensor(
        [math.sqrt(x / math.fsum(probabilities)) for x in probabilities], dtype=dtype
    )
    eye = torch.eye(8, dtype=dtype)
    direction = eye[:, 0] - vector
    prepare = eye - 2 * torch.outer(direction, direction.conj()) / direction.vdot(
        direction
    )
    circuit = fq.Circuit(3, dtype=dtype).any(0, 1, 2, matrix=prepare)
    for which, time, order, repetitions in schedule:
        assert order == "1"
        repetitions = int(repetitions)
        product = eye
        for pauli, coefficient in groups[int(which)]:
            angle = float(time) * coefficient / repetitions
            rotation = math.cos(angle) * eye - 1j * math.sin(angle) * pauli
            product = rotation @ product
        circuit.any(0, 1, 2, matrix=torch.linalg.matrix_power(product, repetitions))
    return circuit


pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def original() -> Path:
    manifest = json.loads((FIXTURES / "aqc_qlsp_sources.json").read_text())
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


def test_original_aqc_preserves_all_200_ordered_steps():
    path = original()
    program = synth.synthesize(path, max_workspace=0)
    actual = fq.run(program).statevector()
    expected = fq.run(aqc_reference(path.read_text())).statevector()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert program.report["workspace_qubits"] == 0
    assert dict(program.registers) == {"qba": (0, 1, 2)}
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(path, max_operations=20)


SMALL = """
qfunc evolve(terms:PauliTerm[], q:qbit[2]) {
  suzuki_trotter(terms, 0.52, 2, 3, q);
  suzuki_trotter(terms, -0.21, 1, 2, q);
}
qfunc main(output q:qbit[2]) {
  allocate(q); X(q[0]); H(q[1]); S(q[1]);
  evolve([
    PauliTerm {pauli=[Pauli::X, Pauli::Z], coefficient=0.4},
    PauliTerm {pauli=[Pauli::Y, Pauli::I], coefficient=0.3},
    PauliTerm {pauli=[Pauli::I, Pauli::I], coefficient=-0.2}
  ], q);
}
"""


def test_order_two_then_order_one_matches_independent_noncommuting_product():
    dtype = torch.complex128
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))
    identity = torch.eye(4, dtype=dtype)
    matrices = [
        (torch.kron(z, x), 0.4),
        (torch.kron(torch.eye(2, dtype=dtype), y), 0.3),
        (identity, -0.2),
    ]
    reference = fq.Circuit(2, dtype=dtype).x(1).h(0).s(0)
    for time, order, count in ((0.52, 2, 3), (-0.21, 1, 2)):
        terms = matrices if order == 1 else matrices + list(reversed(matrices))
        for _ in range(count):
            for matrix, weight in terms:
                angle = time * weight / (order * count)
                reference.any(
                    0,
                    1,
                    matrix=math.cos(angle) * identity - 1j * math.sin(angle) * matrix,
                )
    program = synth.synthesize(SMALL, source_format="qmod", max_workspace=0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-11,
        rtol=0,
    )


def test_dense_pauli_helper_preserves_exact_target_width_checks():
    source = SMALL.replace("Pauli::X, Pauli::Z", "Pauli::X")
    with pytest.raises(InteropConversionError, match="word length must match"):
        synth.synthesize(source, source_format="qmod")
