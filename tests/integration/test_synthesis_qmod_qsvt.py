"""Verify borrowed quantum callbacks and the unmodified Classiq QSVT source."""

import cmath
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
    manifest = json.loads((FIXTURES / "qsvt_sources.json").read_text())
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
    # Qmod arrays are little-endian within each named output register.
    projection, deletion, select, aux = 0, 1, 2, 9
    data = tuple(range(3, 9))
    circuit = fq.Circuit(10, dtype=DTYPE)
    shift = torch.zeros((256, 256), dtype=DTYPE)
    for selector in range(2):
        for word in range(128):
            displacement = 1 if selector == 0 else -1
            shift[
                selector * 128 + (word + displacement) % 128, selector * 128 + word
            ] = 1
    reflection = torch.diag(
        torch.tensor(
            [1 if (column & 64) or not (column & 63) else -1 for column in range(128)],
            dtype=DTYPE,
        )
    )

    def project(angle: float) -> None:
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
            for word in range(16)
        ]
        circuit.any(
            0, 1, 2, aux, matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE))
        )

    def reflect() -> None:
        circuit.h(projection)
        circuit.any(projection, *data, matrix=reflection)
        circuit.h(projection)

    def block_encoding(inverse: bool = False) -> None:
        if inverse:
            reflect()
        circuit.h(select)
        circuit.any(
            select, deletion, *data, matrix=shift.mH.contiguous() if inverse else shift
        )
        circuit.h(select)
        if not inverse:
            reflect()

    circuit.h(aux)
    project(1.280311896404252)
    block_encoding()
    project(8.127145628464149)
    block_encoding(inverse=True)
    project(1.843960321284562)
    block_encoding()
    project(-5.002873410775335)
    circuit.h(aux)
    return circuit


def test_original_qsvt_full_vector_and_clean_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path, memory_limit_bytes=64 * 1024**2)
    actual = fq.run(program).statevector().reshape(1024, -1)
    expected = fq.run(reference()).statevector().reshape(1024)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-11, rtol=0
    )


@pytest.mark.parametrize(
    ("call", "diagnostic"),
    [
        ("invoke(lambda(a,b) {X(a[0]);}, q, q);", "arguments must be disjoint"),
        ("invoke(lambda(a) {X(a[0]);}, q, r);", "two distinct array ports"),
        ("invoke(lambda(a,b) {X(secret[0]);}, q, r);", "qubit array"),
        (
            "invoke(lambda(a,b) {allocate(1,a);}, q, r);",
            "pure unitary effects|allocation",
        ),
    ],
)
def test_borrowed_pair_rejects_bad_callbacks(call: str, diagnostic: str) -> None:
    source = """
        qfunc invoke(u:qfunc(qbit[],qbit[]), x:qbit[], y:qbit[]) {u(x,y);}
        qfunc main(output q:qbit[2], output r:qbit[2], output secret:qbit[2]) {
            allocate(q); allocate(r); allocate(secret);
    """ + call + "}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
