"""Verify the pinned full_hea circuit and static callback ownership."""

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
ANGLES = (
    (0.17, -0.22, 0.31, 0.48, -0.51, 0.63, 0.72, -0.83, 0.96, 0.12, -0.26, 0.37),
    (-0.91, 0.85, -0.72, 0.61, 0.56, -0.47, 0.35, 0.29, -0.16, -0.84, 0.75, 0.64),
)


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "hea_sources.json").read_text())
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


def reference(angles: tuple[float, ...]) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=DTYPE)
    for layer in range(3):
        for index, wire in enumerate((3, 2, 1, 0)):
            circuit.x(wire).ry(wire, theta=angles[4 * layer + index])
        circuit.cnot(3, 2).cnot(2, 1).cnot(1, 0)
    return circuit


@pytest.mark.parametrize("angles", ANGLES)
def test_original_hea_preserves_full_vector_and_parameter_order(angles):
    path, _ = original()
    program = synth.synthesize(path, classical_inputs={"params": angles})
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference(angles)).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0
    resources = program.report["resources"]
    assert resources.n_operations <= 33
    assert resources.operation_counts["cx"] == 9


def test_hea_parameterized_pair_uses_declared_operand_order():
    source = """qfunc main(output x:qbit[2]) {
      allocate(x); H(x[1]);
      full_hea(2, [1,1], [0.19,0.42,0.73], [[0,1]], 1,
        [lambda(a,q) {RY(a,q);}], [lambda(a,c,t) {CRX(a,t,c);}], x);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).h(0).ry(1, theta=0.19).ry(0, theta=0.42)
    reference.crx(0, 1, theta=0.73)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("flags", "angles", "edges", "body", "diagnostic"),
    [
        ("[1,0]", "[0.3]", "[[0,1]]", "RY(a,q);", "exactly 2 angles"),
        ("[1,0]", "[0.3,0.5]", "[[0,2]]", "RY(a,q);", "endpoints"),
        ("[1,0]", "[0.3,0.5]", "[[0,0]]", "RY(a,q);", "endpoints"),
        ("[1,0]", "[0.3,0.5]", "[[0,1]]", "RY(a,other);", "declared qubit"),
        ("[1,0]", "[0.3,0.5]", "[[0,1]]", "RY(0.3,q);", "angle formal"),
        ("[0,0]", "[]", "[[0,1]]", "RY(a,q);", "angle flag"),
    ],
)
def test_hea_rejects_invalid_connectivity_and_lambda_ownership(
    flags, angles, edges, body, diagnostic
):
    source = f"""qfunc main(output x:qbit[2],output other:qbit) {{
      allocate(x); allocate(other);
      full_hea(2, {flags}, {angles}, {edges}, 1,
        [lambda(a,q) {{{body}}}], [lambda(a,c,t) {{CX(c,t);}}], x);
    }}"""
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
