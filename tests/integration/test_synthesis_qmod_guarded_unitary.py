"""Verify coherent matrix selection and the pinned harmonic oscillator."""

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

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "harmonic_oscillator_sources.json").read_text())
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


def reference(source: str) -> fq.Circuit:
    matrices = [
        torch.tensor(ast.literal_eval(match), dtype=DTYPE)
        for match in re.findall(r"unitary\(\s*(\[.*?\])\s*,\s*\w+\s*\)", source, re.S)
    ]
    assert len(matrices) == 14
    enc_sign, enc_zero, enc_one, *remaining = matrices
    *rotations, dec_zero, dec_one, dec_sign = remaining
    circuit = fq.Circuit(5, dtype=DTYPE).ry(0, theta=math.pi / 2)
    circuit.any(4, matrix=enc_sign)
    circuit.any(4, 1, 2, 3, matrix=torch.block_diag(enc_zero, enc_one))
    circuit.any(1, 2, 3, 0, matrix=torch.block_diag(*rotations))
    circuit.any(4, 1, 2, 3, matrix=torch.block_diag(dec_zero, dec_one))
    circuit.any(4, matrix=dec_sign)
    return circuit


def test_original_harmonic_oscillator_full_state_and_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(32, -1)
    expected = fq.run(reference(path.read_text())).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


def test_unitary_branches_preserve_relative_sign_and_selector_order() -> None:
    source = """
        qfunc main(output s:qnum<2>,output c:qbit,output t:qbit) {
            allocate(s); allocate(c); allocate(t);
            hadamard_transform(s); H(c);
            control (not c) {unitary([[0,1],[1,0]], t);}
            else {unitary([[1,0],[0,-1]], t);}
            control (s == 1) {unitary([[-1,0],[0,-1]], t);}
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    identity = torch.eye(2, dtype=DTYPE)
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    circuit = fq.Circuit(4, dtype=DTYPE).h(0).h(1).h(2)
    circuit.any(2, 3, matrix=torch.block_diag(x, z))
    circuit.any(
        0, 1, 3, matrix=torch.block_diag(identity, -identity, identity, identity)
    )
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("width", "guard", "target", "message"),
    [
        (2, 1, "s[0]", "disjoint from condition"),
        (5, 1, "t", "matrix width"),
        (2, 4, "t", "selector range"),
    ],
)
def test_guarded_unitary_rejects_alias_and_bounds(
    width: int, guard: int, target: str, message: str
) -> None:
    source = f"""
        qfunc main(output s:qbit[{width}],output t:qbit) {{
            allocate(s); allocate(t);
            control (s == {guard}) {{unitary([[0,1],[1,0]], {target});}}
        }}
    """
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source, source_format="qmod")
