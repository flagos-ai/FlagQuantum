"""Independent matrix construction of the source's OAA program."""

from __future__ import annotations

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


def _pauli_select(source: str) -> torch.Tensor:
    dtype = torch.complex128
    identity = torch.eye(2, dtype=dtype)
    pauli = {
        "I": identity,
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    select = torch.zeros((16, 16), dtype=dtype)
    for control in range(4):
        expression = (
            rf"qfunc unitaries_{control}_lambda___0_0_expanded___0\([^{{]+\{{(.*?)\n\}}"
        )
        [body] = re.findall(expression, source, re.S)
        entries = re.findall(r"pauli=Pauli::([IXYZ]),\s*index=(\d+)", body)
        assert len(entries) == 2
        terms = {int(index): pauli[axis] for axis, index in entries}
        selector = torch.zeros((4, 4), dtype=dtype)
        selector[control, control] = 1
        select += torch.kron(torch.kron(terms[1], terms[0]), selector)
    return select


def oaa_reference(source: str) -> fq.Circuit:
    dtype = torch.complex128
    select = _pauli_select(source)
    assert "inplace_prepare_state([0.25, 0.25, 0.25, 0.25], 0, block)" in source
    h = torch.tensor([[1, 1], [1, -1]], dtype=dtype) / math.sqrt(2)
    prepare_block = torch.kron(torch.eye(4, dtype=dtype), torch.kron(h, h))
    encode = prepare_block @ select @ prepare_block
    zero = torch.diag(torch.tensor([1, 0, 0, 0], dtype=dtype))
    projection = torch.kron(torch.eye(4, dtype=dtype), zero)
    oracle = torch.eye(16, dtype=dtype) - 2 * projection
    diffuse = encode @ (-oracle) @ encode.mH

    [literal] = re.findall(r"prepare_amplitudes\(\s*(\[.*?\])", source, re.S)
    vector = torch.tensor(ast.literal_eval(literal), dtype=dtype)
    vector /= vector.norm()
    delta = torch.eye(4, dtype=dtype)[:, 0] - vector
    prepare = torch.eye(4, dtype=dtype) - 2 * torch.outer(
        delta, delta.conj()
    ) / delta.vdot(delta)
    [repetitions] = re.findall(
        r"oblivious_amplitude_amplification_expanded___0\((\d+), block, data\)", source
    )
    circuit = (
        fq.Circuit(4, dtype=dtype)
        .any(0, 1, matrix=prepare)
        .any(*range(4), matrix=encode)
    )
    for _ in range(int(repetitions)):
        circuit.any(*range(4), matrix=oracle).any(*range(4), matrix=diffuse)
    return circuit


pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classiq_qmod"


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "oaa_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((_FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path


def test_unchanged_oaa_retains_absolute_phase_and_cleans_marker() -> None:
    source = _original()
    program = synth.synthesize(source, max_qubits=4, max_workspace=0)
    expected = fq.run(oaa_reference(source.read_text())).statevector()
    assert dict(program.registers) == {"data": (0, 1), "block": (2, 3)}
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    ("before", "after", "diagnostic"),
    [
        ("H(target);", "X(target);", "cannot prove annotated borrowed phase view"),
        ("X(aux);", "H(aux);", "cannot prove annotated borrowed phase view"),
        ("index=1", "index=2", "index is out of bounds"),
        (
            "pauli_string[i].pauli",
            "pauli_string[4].pauli",
            "IndexedPauli array index is out of bounds",
        ),
    ],
)
def test_unproved_kickback_and_indexes_are_rejected(
    before: str, after: str, diagnostic: str
) -> None:
    raw = _original().read_text()
    assert before in raw
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(raw.replace(before, after, 1), source_format="qmod")


def test_selected_pauli_preserves_relative_phase_on_all_control_values() -> None:
    original = _original().read_text()
    modified = original.split("qfunc main(", 1)[0] + """
    qfunc main(output data:qbit[2], output block:qnum<2>) {
      allocate(data); allocate(block);
      hadamard_transform(data); hadamard_transform(block);
      S(data[0]); X(data[1]);
      select_0_lambda___0_0_expanded___0(block, data);
    }
    """
    program = synth.synthesize(modified, source_format="qmod", max_workspace=0)
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.s(1).x(0).any(*range(4), matrix=_pauli_select(original))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_static_indexed_pauli_cannot_write_a_const_port() -> None:
    source = """
    qperm bad(ops:IndexedPauli[], const data:qbit[1]) {
      X(data[ops[0].index]);
    }
    qfunc main(output data:qbit[1]) {
      allocate(data); bad([IndexedPauli {pauli=1,index=0}], data);
    }
    """
    with pytest.raises(
        InteropConversionError, match="cannot write a Const quantum port"
    ):
        synth.synthesize(source, source_format="qmod")
