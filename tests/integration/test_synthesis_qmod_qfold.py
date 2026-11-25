"""Verify the original QFold walk and bounded scientific literal controls."""

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


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "qfold_sources.json").read_text())
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


def reference() -> fq.Circuit:
    def rotation(angle: float, mask: int, expected: int) -> torch.Tensor:
        matrix = torch.eye(16, dtype=DTYPE)
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        for word in range(16):
            if not word & 2 and word & mask == expected:
                partner = word | 2
                matrix[word, word] = matrix[partner, partner] = c
                matrix[word, partner], matrix[partner, word] = -s, s
        return matrix

    coin = fq.Circuit(4, dtype=DTYPE).x(2).h(3)
    coin.any(0, 1, 2, 3, matrix=rotation(4.994422698e-6, 9, 0))
    coin.any(0, 1, 2, 3, matrix=rotation(1.372174780775247, 5, 1)).x(2)
    shift = torch.zeros((16, 16), dtype=DTYPE)
    for word in range(16):
        toggle = (4 if word & 1 else 8) if word & 2 else 0
        shift[word ^ toggle, word] = 1
    oracle = torch.diag(
        torch.tensor([-1 if word & 3 == 0 else 1 for word in range(16)], dtype=DTYPE)
    )
    return (
        fq.Circuit(4, dtype=DTYPE)
        .h(0)
        .h(1)
        .compose(coin)
        .any(0, 1, 2, 3, matrix=shift)
        .compose(coin.adjoint())
        .any(0, 1, 2, 3, matrix=oracle)
    )


def test_original_qfold_full_state_and_output_marginal() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    result = fq.run(program)
    actual = result.statevector().reshape(16, -1)
    expected = fq.run(reference()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    marginal = expected.abs().square().reshape(4, 4).sum(dim=1)
    assert dict(
        program.probabilities(result, registers=("phi", "psi"))
    ) == pytest.approx({(i // 2, i % 2): float(p) for i, p in enumerate(marginal)})


@pytest.mark.parametrize("literal", ["5E-1", "0.05e+1"])
def test_scientific_literal_preserves_small_controlled_rotation(literal: str) -> None:
    source = """
        qfunc main(output a:qnum<1>,output b:qbit,output target:qnum<1>) {
            allocate(a);allocate(b);allocate(target);H(a);H(b);
            control ((a == 0) & (b == 1)) {RY(ANGLE,target);}
        }
    """
    program = synth.synthesize(source.replace("ANGLE", literal), source_format="qmod")
    expected = (
        torch.tensor([1, 0, math.cos(0.25), math.sin(0.25), 1, 0, 1, 0], dtype=DTYPE)
        / 2
    )
    actual = fq.run(program).statevector().reshape(8, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("literal", ["1e309", "1e-900", "1e999999999999"])
def test_nonrepresentable_scientific_literals_are_rejected(literal: str) -> None:
    source = f"qfunc main(output x:qbit) {{allocate(x); RY({literal},x);}}"
    with pytest.raises(InteropConversionError, match="scientific literal"):
        synth.synthesize(source, source_format="qmod")


def test_conjunctive_control_rejects_target_alias() -> None:
    source = """
        qfunc main(output a:qnum<1>,output b:qbit) {
            allocate(a);allocate(b);
            control ((a == 0) & (b == 1)) {X(a);}
        }
    """
    with pytest.raises(
        InteropConversionError, match="controls and target must be distinct"
    ):
        synth.synthesize(source, source_format="qmod")


def test_scientific_literals_in_a_probability_vector() -> None:
    source = """
        qfunc main(output target:qbit) {
            allocate(target);
            inplace_prepare_state([2.5e-1, 75E-2], 0e+0, target);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    expected = torch.tensor([0.5, math.sqrt(0.75)], dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )
