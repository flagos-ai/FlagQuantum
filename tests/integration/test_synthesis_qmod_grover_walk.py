"""Verify the original coined walks and phase-exact Grover diffusion."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original(name: str) -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "grover_walk_sources.json").read_text())
    row = next(row for row in manifest["fixtures"] if row["file"] == f"{name}.qmod")
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


def reference(steps: int, hypercube: bool) -> fq.Circuit:
    diffuser = torch.full((4, 4), 0.5, dtype=DTYPE) - torch.eye(4, dtype=DTYPE)
    if hypercube:
        circuit = fq.Circuit(6, dtype=DTYPE).h(4).h(5)
        shift = torch.zeros((64, 64), dtype=DTYPE)
        for position in range(16):
            for coin in range(4):
                shift[4 * (position ^ (1 << coin)) + coin, 4 * position + coin] = 1
        for _ in range(steps):
            circuit.any(4, 5, matrix=diffuser).any(*range(6), matrix=shift)
        return circuit

    circuit = fq.Circuit(12, dtype=DTYPE).x(0).x(5).h(10).z(10).h(11)
    shift_x = torch.zeros((128, 128), dtype=DTYPE)
    shift_y = torch.zeros_like(shift_x)
    for coin, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        for position in range(32):
            shift_x[32 * coin + (position + dx) % 32, 32 * coin + position] = 1
            shift_y[32 * coin + (position + dy) % 32, 32 * coin + position] = 1
    for _ in range(steps):
        circuit.any(10, 11, matrix=diffuser)
        circuit.any(10, 11, *range(5), matrix=shift_x)
        circuit.any(10, 11, *range(5, 10), matrix=shift_y)
    return circuit


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
@pytest.mark.parametrize("hypercube", [False, True])
def test_original_grover_walk_full_state(steps: int, hypercube: bool) -> None:
    name = "quantum_walk_hypercube" if hypercube else "discrete_time_quantum_walk"
    path, _ = original(name)
    program = synth.synthesize(path, classical_inputs={"t": steps})
    result = fq.run(program)
    expected = fq.run(reference(steps, hypercube)).statevector().flatten()
    actual = result.statevector().reshape(expected.numel(), -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    if not hypercube:
        marginal = expected.abs().square().reshape(32, 32, 4).sum(dim=-1)
        reported = dict(program.probabilities(result, registers=("x", "y")))
        for x, y in zip(*torch.where(marginal > 1e-15), strict=True):
            assert reported[int(x), int(y)] == pytest.approx(float(marginal[x, y]))


@pytest.mark.parametrize(
    "body", ["H(a);", "allocate(a);", "hadamard_transform(other);"]
)
def test_diffuser_refuses_unsupported_or_capturing_preparation(body: str) -> None:
    source = (
        "qfunc main(output x:qnum<2>,output other:qnum<2>) {"
        "allocate(x);allocate(other);grover_diffuser(lambda(a){" + body + "},x);}"
    )
    with pytest.raises(InteropConversionError, match="diffusion.preparation"):
        synth.synthesize(source, source_format="qmod")


def test_multibit_selector_rejects_overlapping_helper() -> None:
    source = """
        qfunc shift(x:qnum) {x += 1;}
        qfunc main(output x:qnum<2>) {
            allocate(x);
            control (x == 2) {shift(x);}
        }
    """
    with pytest.raises(InteropConversionError, match="distinct qubit locations"):
        synth.synthesize(source, source_format="qmod")


def test_diffuser_uses_the_absolute_sign_of_a_nonuniform_reflection() -> None:
    source = """
        qfunc main(output x:qbit[2]) {
            allocate(x); X(x[1]);
            grover_diffuser(lambda(a){inplace_prepare_state([0.125,0.25,0.5,0.125],0,a);},x);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    psi = torch.tensor([0.125, 0.25, 0.5, 0.125], dtype=torch.float64).sqrt().to(DTYPE)
    reflection = 2 * torch.outer(psi, psi) - torch.eye(4, dtype=DTYPE)
    reference = fq.Circuit(2, dtype=DTYPE).x(0).any(0, 1, matrix=reflection)
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
