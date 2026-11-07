"""Check affine register updates and the original Toeplitz flag semantics."""

import hashlib
import itertools
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
    manifest = json.loads((FIXTURES / "toeplitz_sources.json").read_text())
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
    circuit = fq.Circuit(8, dtype=DTYPE)
    for wire in range(1, 7):
        circuit.h(wire)
    blocks = []
    for entry in (0.6406, 0.6246, -0.3873, 0.952):
        off = -1j * math.sqrt(1 - entry**2)
        blocks.append(torch.tensor([[entry, off], [off, entry]], dtype=DTYPE))
    circuit.any(1, 2, 0, matrix=torch.block_diag(*blocks))
    perm = torch.zeros(128, 128, dtype=DTYPE)
    for s, j, flag in itertools.product(range(4), range(16), range(2)):
        captured = j + s - 1
        before = (s * 16 + j) * 2 + flag
        after = (s * 16 + captured % 16) * 2 + (flag ^ int(not 0 <= captured < 16))
        perm[after, before] = 1
    circuit.any(*range(1, 8), matrix=perm)
    return circuit.h(1).h(2)


def unitary_columns(circuit, reference_circuit, workspace: int):
    data = torch.eye(256, dtype=DTYPE)
    inputs = torch.zeros(256, 256, 1 << workspace, dtype=DTYPE)
    inputs[:, :, 0] = data
    actual = fq.Circuit.from_ir(
        circuit, dtype=DTYPE, inputs=inputs.flatten(1), bsz=256
    ).state()
    expected = fq.Circuit.from_ir(
        reference_circuit.to_ir(), dtype=DTYPE, inputs=data, bsz=256
    ).state()
    return actual.reshape(256, 256, -1), expected


def test_original_toeplitz_all_columns_and_zero_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    result = fq.run(program)
    assert dict(program.probabilities(result, registers=("dlt",))) == pytest.approx(
        {(0,): 15 / 16, (1,): 1 / 16}
    )
    workspace = program.report["workspace_qubits"]
    actual, expected = unitary_columns(program.circuit, reference(), workspace)
    torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, :, 1:], torch.zeros_like(actual[:, :, 1:]), atol=1e-12, rtol=0
    )


def test_affine_update_with_negative_source_and_offset() -> None:
    source = """
        qfunc update(j:qnum, s:qnum, k:int) {j += k - s;}
        qfunc main(output j:qnum<3>, output s:qnum<2>) {
            allocate(j); allocate(s); apply_to_all(H,s); update(j,s,2);
        }
    """
    p = synth.synthesize(source, source_format="qmod")
    probabilities = dict(p.probabilities(fq.run(p), registers=("j", "s")))
    assert probabilities == pytest.approx({((2 - s) % 8, s): 0.25 for s in range(4)})


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("j += j;", "source and target must be disjoint"),
        (
            "tmp:qnum; tmp = j + s; touch(j,s,dlt); control(tmp > 0) {X(dlt);}",
            "disjoint reversible integer update",
        ),
        (
            "tmp:qnum; tmp = j + s; touch(j,s,dlt); control(tmp > 0) {X(j);}",
            "snapshot inputs and flag must be disjoint",
        ),
    ],
)
def test_snapshot_and_affine_alias_refusals(body: str, diagnostic: str) -> None:
    source = "qfunc touch(j:qnum,s:qnum,dlt:qbit) {X(dlt);}"
    source += "qfunc run(j:qnum,s:qnum,dlt:qbit) {" + body + "}"
    source += "qfunc main(output j:qnum<1>,output s:qnum<2>,output dlt:qbit) {allocate(j);allocate(s);allocate(dlt);run(j,s,dlt);}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
