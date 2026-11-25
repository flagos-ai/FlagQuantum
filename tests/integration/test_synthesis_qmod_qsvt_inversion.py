"""Verify unchanged QSVT inversion against explicit matrix and phase operations."""

import ast
import cmath
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "qsvt_inversion_sources.json").read_text())
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
    return path, row


def source_literals(path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...]]:
    source = path.read_text()
    matrix = re.search(r"unitary\((\[.*?\]), qsvt_state\)", source, re.S)
    rhs = re.search(
        r"inplace_prepare_amplitudes\((\[.*?\]), 0, qsvt_state.state\)", source, re.S
    )
    phases = re.search(r"qsvt_inversion\((\[.*?\]), lambda", source, re.S)
    assert matrix is not None and rhs is not None and phases is not None
    return (
        torch.tensor(ast.literal_eval(matrix[1]), dtype=DTYPE),
        torch.tensor(ast.literal_eval(rhs[1]), dtype=DTYPE),
        tuple(ast.literal_eval(phases[1])),
    )


def reference(
    unitary: torch.Tensor, rhs: torch.Tensor, phases: tuple[float, ...]
) -> fq.Circuit:
    eye = torch.eye(4, dtype=DTYPE)
    delta = eye[:, 0] - rhs / torch.linalg.vector_norm(rhs)
    preparation = eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    circuit = fq.Circuit(4, dtype=DTYPE).any(1, 2, matrix=preparation).h(3)
    # The block field is most significant and the QSVT auxiliary is least significant.
    signs = [1 if bool(word & 1) != (word < 8) else -1 for word in range(16)]

    def phase(angle: float) -> torch.Tensor:
        return torch.diag(
            torch.tensor(
                [cmath.exp(0.5j * angle * sign) for sign in signs], dtype=DTYPE
            )
        )

    circuit.any(0, 1, 2, 3, matrix=phase(phases[0]))
    for angle_inverse, angle_forward in zip(phases[1:-1:2], phases[2::2], strict=True):
        circuit.any(0, 1, 2, matrix=unitary.mH)
        circuit.any(0, 1, 2, 3, matrix=phase(angle_inverse))
        circuit.any(0, 1, 2, matrix=unitary)
        circuit.any(0, 1, 2, 3, matrix=phase(angle_forward))
    return (
        circuit.any(0, 1, 2, matrix=unitary.mH)
        .any(0, 1, 2, 3, matrix=phase(phases[-1]))
        .h(3)
    )


def test_original_qsvt_inversion_full_state_and_linear_solution() -> None:
    path, _ = original()
    unitary, rhs, phases = source_literals(path)
    assert len(phases) == 70
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(16, -1)
    expected = fq.run(reference(unitary, rhs, phases)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-10, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    matrix = unitary[:4, :4]
    solution = torch.linalg.solve(matrix, rhs) * torch.linalg.svdvals(matrix)[-1] / 2
    selected = actual[:, 0].reshape(2, 4, 2)[0, :, 0]
    torch.testing.assert_close(selected, solution, atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    ("projector", "block", "phases", "diagnostic"),
    [
        (
            "lambda(m){m ^= q == 0;}",
            "lambda(){X(q);}",
            "[0.2,0.3,0.4]",
            "even phase count",
        ),
        ("lambda(m){H(m);}", "lambda(){X(q);}", "[0.2,0.3]", "may only XOR"),
        (
            "lambda(m){m ^= aux == 0;}",
            "lambda(){X(q);}",
            "[0.2,0.3]",
            "cannot read its XOR target",
        ),
        (
            "lambda(m){m ^= q == 0;}",
            "lambda(){X(aux);}",
            "[0.2,0.3]",
            "must be disjoint",
        ),
        (
            "lambda(m){m ^= q == 0;}",
            "lambda(){allocate(q);}",
            "[0.2,0.3]",
            "pure effects",
        ),
    ],
)
def test_qsvt_inversion_refuses_invalid_projector_or_block(
    projector: str, block: str, phases: str, diagnostic: str
) -> None:
    source = (
        "qfunc main(output q:qbit,output aux:qbit) {allocate(q);allocate(aux);"
        f"qsvt_inversion({phases},{projector},{block},aux);}}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
