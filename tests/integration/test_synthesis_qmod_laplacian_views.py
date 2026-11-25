"""Verify the pinned two-axis Laplacian and nested numeric-view controls."""

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
    manifest = json.loads((FIXTURES / "laplacian_view_sources.json").read_text())
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


def prepare() -> torch.Tensor:
    mass = torch.tensor([0.2, 0, 0.2, 0.2, 0.2, 0.2, 0, 0], dtype=torch.float64)
    transform = torch.eye(8, dtype=DTYPE)
    for stride in (4, 2, 1):
        layer = torch.zeros_like(transform)
        for group in range(0, 8, 2 * stride):
            left = float(mass[group : group + stride].sum())
            right = float(mass[group + stride : group + 2 * stride].sum())
            angle = math.atan2(math.sqrt(right), math.sqrt(left))
            for i in range(stride):
                a, b = group + i, group + i + stride
                layer[a, a] = layer[b, b] = math.cos(angle)
                layer[b, a] = math.sin(angle)
                layer[a, b] = -math.sin(angle)
        transform = layer @ transform
    return transform


def reference() -> fq.Circuit:
    circuit = fq.Circuit(9, dtype=DTYPE)
    prep = prepare()
    for i in range(3, 7):
        circuit.h(i)
    circuit.any(0, 1, 2, matrix=prep)
    before = torch.zeros((256, 256), dtype=DTYPE)
    after = torch.zeros((128, 128), dtype=DTYPE)
    rotations = []
    for state in range(8):
        b0, b1, b2 = ((state >> i) & 1 for i in range(3))
        index = b1 + 2 * b2
        cosine = (-1.0, 0.25, 0.25, 1.0)[index]
        sine = math.sqrt(1 - cosine * cosine)
        rotations.append(
            torch.tensor([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=DTYPE)
        )
        for j in range(16):
            first = (j + (4 if not b0 and b2 else 0)) % 16
            first = (first + int(bool(b1 and not b0))) % 16
            flip = (
                (state == 1)
                ^ (bool(b1) and first % 4 == 0)
                ^ (bool(b2) and first // 4 == 0)
            )
            for bit in range(2):
                before[
                    (state * 16 + first) * 2 + (bit ^ flip), (state * 16 + j) * 2 + bit
                ] = 1
            final0 = b0 ^ b1 ^ b2
            last = (j - int(bool(b1 and not final0))) % 16
            last = (last - (4 if b2 and not final0 else 0)) % 16
            after[(state ^ b0 ^ final0) * 16 + last, state * 16 + j] = 1
    circuit.any(*range(7), 8, matrix=before)
    circuit.z(7)
    circuit.any(0, 1, 2, 7, matrix=torch.block_diag(*rotations))
    circuit.any(*range(7), matrix=after)
    circuit.any(0, 1, 2, matrix=prep.mH.contiguous())
    return circuit


def test_original_two_axis_laplacian_full_state_and_clean_workspace() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(512, -1)
    expected = fq.run(reference()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


def test_nested_numeric_control_rejects_alias() -> None:
    source = """
        qfunc main(output s:qnum<1>,output t:qnum<1>) {
            allocate(s); allocate(t);
            control (s == 1) { control (t == 0) { X(s); } }
        }
    """
    with pytest.raises(InteropConversionError, match="distinct qubit locations"):
        synth.synthesize(source, source_format="qmod")


def test_nested_numeric_views_match_mixed_polarity_on_superposition() -> None:
    source = """
        qfunc main(output q:qnum<2>,output t:qnum<1>) {
            allocate(q); allocate(t); hadamard_transform(q);
            hi:qnum<1>; lo:qnum<1>;
            q -> {lo,hi};
            control (lo == 1) { control (hi == 0) { X(t); } }
            {lo,hi} -> q;
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    circuit = fq.Circuit(3, dtype=DTYPE).h(0).h(1).x(0).ccx(0, 1, 2).x(0)
    actual = fq.run(program).statevector().reshape(8, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
