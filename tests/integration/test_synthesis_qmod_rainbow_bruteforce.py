"""Independent full-state Circuit for the pinned Rainbow brute-force sources."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize


def rainbow_reference(source: str, repetitions: int) -> fq.Circuit:
    probabilities = [
        ast.literal_eval(text)
        for text in re.findall(r"inplace_prepare_state\(\s*(\[.*?\])", source, re.S)
    ]
    [angle_text] = re.findall(r"select_rotation\(Pauli::Y,\s*(\[.*?\])", source, re.S)
    angles = ast.literal_eval(angle_text)
    assert len(probabilities) == 2 and all(len(table) == 4 for table in probabilities)
    assert len(angles) == 8
    assert "res = max(x1, (0.5 * (x1 + x2)) + (-1.5));" in source

    dtype = torch.complex128
    prepared = torch.zeros(32, dtype=dtype)
    norms = tuple(math.fsum(table) for table in probabilities)
    for x1 in range(4):
        for x2 in range(4):
            result = max(Fraction(x1), Fraction(x1 + x2 - 3, 2))
            raw = result * 2
            assert raw.denominator == 1 and 0 <= raw < len(angles)
            angle = angles[int(raw)]
            scale = math.sqrt(
                (probabilities[0][x1] / norms[0]) * (probabilities[1][x2] / norms[1])
            )
            index = x1 + 4 * x2
            prepared[index] = scale * math.cos(angle / 2)
            prepared[16 + index] = scale * math.sin(angle / 2)

    identity = torch.eye(32, dtype=dtype)
    difference = identity[:, 0] - prepared
    prepare = identity - 2 * torch.outer(
        difference, difference.conj()
    ) / difference.vdot(difference)
    diffuse = 2 * torch.outer(prepared, prepared.conj()) - identity
    circuit = fq.Circuit(5, dtype=dtype).any(*range(5), matrix=prepare)
    for _ in range(repetitions):
        circuit.z(0).any(*range(5), matrix=diffuse)
    return circuit


pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classiq_qmod"


def _originals() -> tuple[Path, ...]:
    manifest = json.loads((_FIXTURES / "rainbow_bruteforce_sources.json").read_text())
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    paths = []
    for row in manifest["fixtures"]:
        path = _FIXTURES / row["file"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert (
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            == row["git_blob_sha1"]
        )
        paths.append(path)
    return tuple(paths)


@pytest.mark.parametrize("repetitions", [0, 1, 2])
@pytest.mark.parametrize("source_index", [0, 1])
def test_rainbow_original_matches_every_amplitude(
    repetitions: int, source_index: int
) -> None:
    path = _originals()[source_index]
    program = _synthesize(
        path, classical_inputs={"k": repetitions}, max_workspace=0, max_qubits=5
    )
    reference = rainbow_reference(path.read_text(), repetitions)
    assert dict(program.registers) == {"indicator": (0,)}
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_operations == (18, 354, 690)[repetitions]
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-11,
        rtol=0,
    )


PROGRAM = """
qperm encode(const a:qnum<2>, const b:qnum<2>, output score:qnum<3,False,1>) {
    score = (0.5 * a) + (0.5 * b);
}
qfunc select(const bits:qbit[3], target:qbit) {
    select_rotation(Pauli::Z,[0.31,0.37,-0.1,0.62,1.33,0.09,0.84,1.1],bits,target);
}
qfunc rotate(const score:qnum<3,False,1>, target:qbit) {
    raw:qnum<3,False,3>;
    score -> raw;
    select(raw,target);
    raw -> score;
}
qfunc compute_then_rotate(a:qnum<2>, b:qnum<2>, target:qbit) {
    score:qnum<3,False,1>;
    encode(a,b,score);
    rotate(score,target);
}
qfunc main(output a:qnum<2>, output b:qnum<2>, output target:qbit) {
    allocate(a); allocate(b); allocate(target);
    hadamard_transform(a); hadamard_transform(b); H(target); S(target);
    compute_then_rotate(a,b,target);
}
"""


def test_independent_inputs_and_complex_coherence() -> None:
    program = synth.synthesize(PROGRAM, source_format="qmod", max_workspace=0)
    dtype = torch.complex128
    reference = fq.Circuit(5, dtype=dtype)
    for wire in range(5):
        reference.h(wire)
    reference.s(4)
    angles = (0.31, 0.37, -0.1, 0.62, 1.33, 0.09, 0.84, 1.1)
    table = torch.tensor(
        [
            sign * angles[a + b] / 2
            for a in range(4)
            for b in range(4)
            for sign in (-1, 1)
        ],
        dtype=torch.float64,
    )
    reference.any(*range(5), matrix=torch.diag(torch.exp(1j * table)))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("(0.5 * a) + (0.5 * b)", "0.25 * a", "not exactly representable"),
        ("(0.5 * a) + (0.5 * b)", "a + b", "not exactly representable"),
        ("encode(a,b,score)", "encode(a,a,score)", "must be disjoint"),
        ("raw -> score;", "X(target);", "checked computed rotation"),
        (
            "score:qnum<3,False,1>;",
            "score:qnum<3,False,2>;",
            "matching width and scale",
        ),
        (
            "rotate(score,target);",
            "rotate(score,target); X(target);",
            "checked computed rotation",
        ),
    ],
)
def test_computed_rotation_refuses_unproved_contract(
    old: str, new: str, diagnostic: str
) -> None:
    assert old in PROGRAM
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(PROGRAM.replace(old, new, 1), source_format="qmod")


def test_constant_composed_table_uses_one_target_rotation() -> None:
    program = synth.synthesize(
        PROGRAM.replace("(0.5 * a) + (0.5 * b)", "1.0"),
        source_format="qmod",
        max_workspace=0,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.s(4).rz(4, theta=-0.1)
    assert program.report["resources"].operation_counts.get("cx", 0) == 0
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_eliminated_selector_still_requires_initialization() -> None:
    source = PROGRAM.replace("(0.5 * a) + (0.5 * b)", "1.0")
    source = source.replace("allocate(a); ", "").replace("hadamard_transform(a); ", "")
    source = source.replace(
        "compute_then_rotate(a,b,target);",
        "compute_then_rotate(a,b,target); allocate(a);",
    )
    with pytest.raises(InteropConversionError, match="uninitialized"):
        synth.synthesize(source, source_format="qmod")
