"""Validate original reversible preparation and coherent Fourier selection."""

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def fourier_select() -> torch.Tensor:
    coordinates = torch.arange(4, dtype=torch.float64)
    fourier = torch.exp(0.5j * math.pi * torch.outer(coordinates, coordinates)) / 2
    identity = torch.eye(4, dtype=DTYPE)
    return torch.block_diag(identity, fourier, fourier.mH, identity)


def rotation(low: float, high: float) -> torch.Tensor:
    norm = math.hypot(low, high)
    if norm == 0:
        return torch.eye(2, dtype=DTYPE)
    return torch.tensor([[low, -high], [high, low]], dtype=DTYPE) / norm


def selector_preparation(source: str) -> tuple[torch.Tensor, tuple[float, ...]]:
    values = tuple(
        ast.literal_eval(
            source.split("lcu_amplitudes: real[] = ", 1)[1].split(";", 1)[0]
        )
    )
    a, b, c, d = (math.sqrt(value) for value in values)
    high = torch.kron(
        rotation(math.hypot(a, b), math.hypot(c, d)), torch.eye(2, dtype=DTYPE)
    )
    low = torch.block_diag(rotation(a, b), rotation(c, d))
    return low @ high, values


def test_original_lcu_matches_dense_selection_and_weighted_zero_branch() -> None:
    manifest = json.loads((FIXTURES / "reversible_scope_sources.json").read_text())
    original = manifest["fixtures"][0]
    path = FIXTURES / original["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == original["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == original["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )

    prepare, probabilities = selector_preparation(raw.decode())
    select = fourier_select()
    expected = (
        fq.Circuit(4, dtype=DTYPE)
        .any(0, 1, matrix=prepare)
        .any(0, 1, 2, 3, matrix=select)
        .any(0, 1, matrix=prepare.mH)
        .state()
    )
    program = synth.synthesize(path)
    actual = fq.run(program).statevector()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    expected_zero = sum(
        weight * select[4 * branch : 4 * branch + 4, 4 * branch]
        for branch, weight in enumerate(probabilities)
    )
    torch.testing.assert_close(
        actual.reshape(4, 4)[0], expected_zero, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


def test_mask_select_distinguishes_fourier_and_adjoint_on_nonzero_data() -> None:
    program = synth.synthesize(
        """
        qfunc main(output selector:qnum<2>, output data:qnum<2>) {
            allocate(selector); allocate(data);
            hadamard_transform(selector); data ^= 1;
            control (selector == 1) { qft(data); }
            control (selector == 2) { invert { qft(data); } }
        }
        """,
        source_format="qmod",
    )
    reference = fq.Circuit(4, dtype=DTYPE).h(0).h(1).x(3)
    reference.any(0, 1, 2, 3, matrix=fourier_select())
    torch.testing.assert_close(
        fq.run(program).statevector(), reference.state(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ("within { S(q); } apply { allocate(q); }", "pure effects on existing storage"),
        ("control (target == 1) { qft(target); }", "disjoint writable target"),
        ("control (q == 4) { qft(target); }", "out of range"),
    ],
)
def test_scope_and_fourier_selection_refuse_invalid_effects(
    body: str, message: str
) -> None:
    source = (
        "qfunc main(output q:qbit, output target:qnum<2>) {"
        "allocate(q);allocate(target);" + body + "}"
    )
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source, source_format="qmod")
