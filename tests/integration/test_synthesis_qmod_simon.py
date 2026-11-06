"""Verify Simon's checked Output callback and its source oracle."""

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


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "simon_sources.json").read_text())
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


def reference() -> tuple[fq.Circuit, torch.Tensor]:
    circuit = fq.Circuit(10, dtype=DTYPE)
    permutation = torch.zeros((1024, 1024), dtype=DTYPE)
    for value in range(32):
        image = min(value, value ^ 6)
        for result in range(32):
            column = 32 * value + result
            permutation[32 * value + (result ^ image), column] = 1
    for wire in range(5):
        circuit.h(wire)
    circuit.any(*range(10), matrix=permutation)
    for wire in range(5):
        circuit.h(wire)

    expected = torch.zeros((32, 32), dtype=DTYPE)
    for value in range(32):
        result = min(value, value ^ 6)
        for phase in range(32):
            sign = (-1) ** ((value & phase).bit_count() % 2)
            expected[phase, result] += sign / 32
    return circuit, expected.flatten()


def test_original_simon_full_state_and_hidden_mask() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().flatten()
    circuit, expected = reference()
    torch.testing.assert_close(
        actual, fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == 0
    marginal = actual.reshape(32, 32).abs().square().sum(dim=1)
    for phase, probability in enumerate(marginal.tolist()):
        assert probability == pytest.approx(
            1 / 16 if (phase & 6).bit_count() % 2 == 0 else 0
        )


WRAPPER = """
qfunc evaluate(f:qfunc(qnum,output qnum), a:qnum, output b:qnum) {
    within {hadamard_transform(a);} apply {f(a,b);}
}
"""


def test_named_initializer_has_one_fresh_output() -> None:
    source = WRAPPER + """
        qperm evaluate_oracle(const r:qnum, output v:qnum) {v = r;}
        qfunc main(output a:qnum<2>,output b:qnum) {
            allocate(a); evaluate(evaluate_oracle,a,b);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(4, dtype=DTYPE).h(0).h(1).cnot(0, 2).cnot(1, 3).h(0).h(1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("oracle", "call", "diagnostic"),
    [
        ("v = r;", "evaluate(evaluate_oracle,a,a);", "disjoint"),
        (
            "v = r;",
            "allocate(2,b); evaluate(evaluate_oracle,a,b);",
            "fresh|initialized",
        ),
        ("v = v ^ r;", "evaluate(evaluate_oracle,a,b);", "cannot read its Output"),
        (
            "v = r;",
            "evaluate(lambda(x,y) {evaluate_oracle(y,x);},a,b);",
            "must forward",
        ),
        ("v = r; X(r);", "evaluate(evaluate_oracle,a,b);", "checked assignment"),
    ],
)
def test_output_initializer_rejects_alias_escape_and_double_init(
    oracle: str, call: str, diagnostic: str
) -> None:
    source = WRAPPER + f"""
        qperm evaluate_oracle(const r:qnum, output v:qnum) {{{oracle}}}
        qfunc main(output a:qnum<2>,output b:qnum) {{allocate(a);{call}}}
    """
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
