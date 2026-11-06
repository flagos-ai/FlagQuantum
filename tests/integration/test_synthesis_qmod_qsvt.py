"""Verify borrowed quantum callbacks and the unmodified Classiq QSVT source."""

import cmath
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


def original(name: str = "classiq_qsvt") -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "qsvt_sources.json").read_text())
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


def heat_reference() -> fq.Circuit:
    deletion, rotation, high, low, aux = 0, 1, 2, 3, 7
    data = (4, 5, 6)
    circuit = fq.Circuit(8, dtype=DTYPE)

    def ry(angle: float) -> torch.Tensor:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)

    identity = torch.eye(2, dtype=DTYPE)
    prepare_select = torch.block_diag(ry(math.pi / 2), identity) @ torch.kron(
        ry(2 * math.atan2(1, math.sqrt(2))), identity
    )
    rotate = torch.block_diag(
        ry(3.252761000151628), identity, ry(3.252761000151628), identity
    )
    shift = torch.zeros((64, 64), dtype=DTYPE)
    for selector in range(4):
        for word in range(16):
            displacement = (selector if selector < 3 else 0) - 1
            shift[16 * selector + (word + displacement) % 16, 16 * selector + word] = 1

    def block_encoding(inverse: bool = False) -> None:
        circuit.any(high, low, matrix=prepare_select)
        if inverse:
            circuit.any(high, low, deletion, *data, matrix=shift.mH.contiguous())
            circuit.any(high, low, rotation, matrix=rotate.mH.contiguous())
        else:
            circuit.any(high, low, rotation, matrix=rotate)
            circuit.any(high, low, deletion, *data, matrix=shift)
        circuit.any(high, low, matrix=prepare_select.mH.contiguous())

    def project(angle: float) -> None:
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
            for word in range(32)
        ]
        circuit.any(
            *range(4), aux, matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE))
        )

    amplitudes = torch.tensor(
        [
            0,
            0.23192061392433,
            0.417906505941275,
            0.521120889169602,
            0.521120889169602,
            0.417906505941275,
            0.23192061392433,
            0,
        ],
        dtype=DTYPE,
    )
    amplitudes = amplitudes / torch.linalg.vector_norm(amplitudes)
    difference = torch.eye(8, dtype=DTYPE)[:, 0] - amplitudes
    prep = torch.eye(8, dtype=DTYPE) - 2 * torch.outer(
        difference, difference
    ) / difference.dot(difference)
    circuit.any(*data, matrix=prep)
    circuit.h(aux)
    project(-1.738078501179465)
    block_encoding()
    project(0.654311402219197)
    block_encoding(inverse=True)
    project(0.654311402219197)
    circuit.h(aux)
    return circuit


def test_original_heat_qsvt_full_vector_and_clean_workspace() -> None:
    path, _ = original("heat_eq_qsvt")
    program = synth.synthesize(path, memory_limit_bytes=64 * 1024**2)
    actual = fq.run(program).statevector().reshape(256, -1)
    expected = fq.run(heat_reference()).statevector().reshape(256)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-11, rtol=0
    )


CALLBACK_SOURCE = """
    qfunc choose(actions:qfunc[](), selector:qnum) {
        repeat(i:actions.len) { control(selector == i) {actions[i]();} }
    }
    qfunc main(output selector:qbit[2.0], output target:qbit) {
        allocate(2.0,selector);
        allocate(target);
        hadamard_transform(selector);
        choose(CALLBACKS,selector);
    }
"""


def test_static_callback_selection_preserves_coherent_inactive_branches() -> None:
    source = CALLBACK_SOURCE.replace(
        "CALLBACKS", "[lambda(){RY(0.7,target);}, lambda(){X(target);}]"
    )
    program = synth.synthesize(source, source_format="qmod")
    actual = fq.run(program).statevector().reshape(8, -1)
    reference = fq.Circuit(3, dtype=DTYPE).h(0).h(1)
    c, s = math.cos(0.35), math.sin(0.35)
    reference.any(
        0,
        1,
        2,
        matrix=torch.block_diag(
            torch.tensor([[c, -s], [s, c]], dtype=DTYPE),
            torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
            torch.eye(4, dtype=DTYPE),
        ),
    )
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("replacement", "bad_value", "diagnostic"),
    [
        ("CALLBACKS", "[lambda(){X(selector[1]);}]", "disjoint from its selector"),
        (
            "CALLBACKS",
            "[lambda(){allocate(target);}]",
            "pure effects on existing storage",
        ),
        ("CALLBACKS", "[lambda(a){X(target);}]", "zero-argument lambdas"),
        ("CALLBACKS", "[lambda(){X(target);},0]", "expected 'lambda'"),
        ("actions[i]()", "actions[i+1]()", "index is out of bounds"),
        ("qbit[2.0]", "qbit[2.1]", "decimal width must be an exact integer"),
        ("allocate(2.0,selector)", "allocate(2.1,selector)", "allocation size"),
    ],
)
def test_static_callback_selection_refusals(
    replacement: str, bad_value: str, diagnostic: str
) -> None:
    source = CALLBACK_SOURCE.replace(replacement, bad_value).replace(
        "CALLBACKS", "[lambda(){X(target);}]"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
