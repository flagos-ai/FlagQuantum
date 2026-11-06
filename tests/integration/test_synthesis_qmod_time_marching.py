"""Verify nested-record time marching and guarded field updates."""

import ast
import cmath
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
    manifest = json.loads((FIXTURES / "time_marching_sources.json").read_text())
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
    arrays = re.search(
        r"qsvt_lcu_expanded___0\(\s*(\[.*?\])\s*,\s*(\[.*?\])", source, re.S
    )
    assert arrays is not None
    even, odd = (ast.literal_eval(group) for group in arrays.groups())
    assert len(even) == 7 and len(odd) == 8
    circuit = fq.Circuit(11, dtype=DTYPE).h(9).h(10).x(0)
    identity = torch.eye(32, dtype=DTYPE)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    time_basis = torch.kron(torch.kron(hadamard, hadamard), torch.eye(8, dtype=DTYPE))
    data = (7, 8, 9, 10, 6)

    def projector(even_angle: float | None, odd_angle: float) -> None:
        diagonal = []
        for lcu in range(2):
            angle = odd_angle if lcu else (even_angle or 0.0)
            for aux in range(2):
                for block in range(2):
                    for time in range(4):
                        marker = aux ^ (block == 0 and time == 0)
                        diagonal.append(cmath.exp(1j * angle * (marker - 0.5)))
        circuit.any(
            4, 5, 6, 7, 8, matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE))
        )

    decrement = torch.zeros((256, 256), dtype=DTYPE)
    for counter in range(8):
        for guard_bits in range(32):
            target = (counter - (guard_bits == 0)) % 8
            decrement[target * 32 + guard_bits, counter * 32 + guard_bits] = 1
    for step in range(4):
        a, b = step / 2, (step + 1) / 2
        rotations = []
        for time in range(4):
            for index in range(4):
                angle = a + (b - a) * time / 4 + index
                co, si = math.cos(angle), math.sin(angle)
                rotations.append(torch.tensor([[co, -si], [si, co]], dtype=DTYPE))
        evolution = time_basis @ torch.block_diag(*rotations) @ time_basis
        circuit.ry(4, theta=math.pi / 2).h(5)
        projector(even[0], odd[0])
        for i in (1, 3, 5):
            circuit.any(*data, matrix=evolution)
            projector(even[i], odd[i])
            circuit.any(*data, matrix=evolution.mH.contiguous())
            projector(even[i + 1], odd[i + 1])
        circuit.any(4, *data, matrix=torch.block_diag(identity, evolution))
        projector(None, odd[-1])
        circuit.h(5).ry(4, theta=-math.pi / 2)
        circuit.any(0, 1, 2, 4, 5, 6, 7, 8, matrix=decrement)
    return circuit


def test_original_time_marching_preserves_four_slices_and_clean_workspace():
    path, _ = original()
    program = synth.synthesize(path)
    actual = fq.run(program).statevector().reshape(2048, -1)
    expected = fq.run(reference(path.read_text())).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 4
    assert program.registers["qbe.counter"] == (0, 1, 2)
    assert program.registers["qbe.time_slice.magnus.time_dependent.index"] == (9, 10)


FIELD_SOURCE = """qstruct State {inner:Flags; count:qnum<2>;}
qstruct Flags {a:qbit; b:qbit;}
qfunc main(output p:State) {
    allocate(p); hadamard_transform(p);
    control ((not p.inner.a) and (p.inner.b == 1)) {p.count += -1;}
}"""


def test_guarded_field_decrement_is_modular_and_coherent():
    circuit = fq.Circuit(4, dtype=DTYPE)
    for wire in range(4):
        circuit.h(wire)
    # Give each basis state a phase so the permutation is visible on the full state.
    phases = torch.tensor([cmath.exp(0.29j * k) for k in range(16)], dtype=DTYPE)
    circuit.any(*range(4), matrix=torch.diag(phases))
    matrix = torch.zeros((16, 16), dtype=DTYPE)
    for basis in range(16):
        flags = basis & 3
        count = basis >> 2
        mapped = (count - (flags == 2)) % 4
        matrix[4 * mapped + flags, basis] = 1
    circuit.any(*range(4), matrix=matrix)
    marked = FIELD_SOURCE.replace(
        "hadamard_transform(p);",
        "hadamard_transform(p); phase(4*p.count+2*p.inner.b+p.inner.a,0.29);",
    )
    program = synth.synthesize(marked, source_format="qmod")
    actual = fq.run(program).statevector().reshape(16, -1)
    torch.testing.assert_close(
        actual[:, 0], fq.run(circuit).statevector().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


def test_guarded_field_update_refuses_control_overlap():
    source = FIELD_SOURCE.replace("not p.inner.a", "p.count == 0")
    with pytest.raises(
        InteropConversionError, match="field controls and target must be disjoint"
    ):
        synth.synthesize(source, source_format="qmod")
