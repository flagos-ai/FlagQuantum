"""Verify the original Value-at-Risk amplification against a rank-one reflection."""

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
    manifest = json.loads((FIXTURES / "value_at_risk_sources.json").read_text())
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


def reference(source: Path, iterations: int) -> fq.Circuit:
    match = re.search(r"inplace_prepare_state\(\[([^\]]+)\]", source.read_text())
    assert match is not None
    weights = [float(item) for item in match[1].split(",")]
    assert len(weights) == 128
    initial = torch.zeros(256, dtype=DTYPE)
    for word, probability in enumerate(weights):
        initial[word + (128 if word < 32 else 0)] = probability**0.5
    initial /= torch.linalg.vector_norm(initial)
    eye = torch.eye(256, dtype=DTYPE)
    delta = eye[:, 0] - initial
    preparation = eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    diffusion = 2 * torch.outer(initial, initial.conj()) - eye
    circuit = fq.Circuit(8, dtype=DTYPE).any(*range(8), matrix=preparation)
    for _ in range(iterations):
        circuit.z(0).any(*range(8), matrix=diffusion)
    return circuit


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_original_value_at_risk_full_state_and_indicator(iterations: int) -> None:
    path, _ = original()
    program = synth.synthesize(path, classical_inputs={"k": iterations})
    result = fq.run(program)
    actual = result.statevector().reshape(256, -1)
    expected = fq.run(reference(path, iterations)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-11, rtol=0
    )
    probabilities = expected.abs().square().reshape(2, 128).sum(dim=1)
    assert dict(
        program.probabilities(result, registers=("indicator",))
    ) == pytest.approx({(i,): float(p) for i, p in enumerate(probabilities)})


def test_unsized_const_predicate_xors_a_nonzero_coherent_marker() -> None:
    source = """
        qperm less(const data:qnum, marker:qbit) {marker ^= data < 2;}
        qfunc use(data:qbit[],marker:qbit) {less(data,marker);}
        qfunc main(output data:qbit[2],output marker:qbit) {
            allocate(data);allocate(marker);hadamard_transform(data);X(marker);
            use(data,marker);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    actual = fq.run(program).statevector().reshape(8, -1)
    expected = torch.tensor([0.5, 0, 0.5, 0, 0, 0.5, 0, 0.5], dtype=DTYPE)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)


def test_unsized_const_without_predicate_proof_refuses() -> None:
    source = """
        qfunc bad(const data:qnum) {H(data);}
        qfunc main(output data:qnum<1>) {allocate(data);bad(data);}
    """
    with pytest.raises(InteropConversionError, match="Const qnum requires"):
        synth.synthesize(source, source_format="qmod")


def test_numeric_predicate_does_not_truncate_a_borrowed_array() -> None:
    source = """
        qperm less(const data:qnum<2>,marker:qbit) {marker ^= data < 2;}
        qfunc main(output data:qbit[3],output marker:qbit) {
            allocate(data);allocate(marker);less(data,marker);
        }
    """
    with pytest.raises(
        InteropConversionError, match="matching nominal record or array width"
    ):
        synth.synthesize(source, source_format="qmod")
