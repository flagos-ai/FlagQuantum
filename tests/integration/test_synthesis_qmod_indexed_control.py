"""Compare indexed Qmod controls with independent phase-sensitive circuits."""

import ast
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


def compression_reference(source: str) -> fq.Circuit:
    matrix_a, matrix_b = (
        torch.tensor(
            ast.literal_eval(piece.split(", {q[0]", 1)[0]), dtype=torch.complex128
        )
        for piece in source.split("unitary(")[1:]
    )
    return (
        fq.Circuit(3, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .y(2)
        .cx(1, 2)
        .ch(2, 1)
        .any(0, 1, 2, matrix=matrix_a)
        .any(0, 1, 2, matrix=matrix_b)
        .cx(0, 1)
        .cx(0, 1)
        .cx(0, 2)
        .ccx(2, 1, 0)
    )


def tutorial_reference(source: str) -> fq.Circuit:
    assert "control (msb == 1)" in source and "RY(pi / 3, lsb)" in source
    matrix = torch.eye(8, dtype=torch.complex128)
    cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
    matrix[2:4, 2:4] = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=torch.complex128
    )
    return (
        fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).any(0, 1, 2, matrix=matrix)
    )


REFERENCES = {
    "quantum_compression_algorithm_for_symmetric_states.qmod": compression_reference,
    "Qmod_tutorial_part2.qmod": tutorial_reference,
}


@pytest.mark.parametrize("name", REFERENCES)
def test_original_program_matches_independent_dense_reference(name: str) -> None:
    manifest = json.loads((FIXTURES / "indexed_control_sources.json").read_text())
    original = next(row for row in manifest["fixtures"] if row["file"] == name)
    path = FIXTURES / original["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == original["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == original["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )

    program = synth.synthesize(path)
    reference = REFERENCES[name](raw.decode())
    torch.testing.assert_close(
        fq.run(program).statevector(), reference.state(), atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


def test_conjunction_control_h_retains_relative_phase() -> None:
    source = """
        qfunc main(output x: qbit[3]) {
            allocate(3, x);
            H(x[0]); H(x[1]); H(x[2]); S(x[2]);
            control (x[0] and x[1]) { H(x[2]); }
        }
    """
    matrix = torch.eye(8, dtype=torch.complex128)
    matrix[6:, 6:] = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2**0.5
    reference = fq.Circuit(3, dtype=torch.complex128).h(2).h(1).h(0).s(0)
    reference.any(2, 1, 0, matrix=matrix)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(), reference.state(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "condition, target, message",
    [
        ("x[0] and x[0]", "x[2]", "distinct qubit locations"),
        ("x[0] and x[1]", "x[1]", "distinct qubit locations"),
        ("x[3]", "x[2]", "index is out of bounds"),
    ],
)
def test_indexed_control_checks_locations(
    condition: str, target: str, message: str
) -> None:
    source = f"qfunc main(output x: qbit[3]) {{ allocate(3,x); control ({condition}) {{ H({target}); }} }}"
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "transfer, diagnostic",
    [
        ("{msb, lsb} -> res;", "original order"),
        ("lsb -> res;", "whole local array"),
        ("{lsb, msb} -> res; H(x[0]);", "suspended"),
    ],
)
def test_output_binding_requires_complete_unique_storage(
    transfer: str, diagnostic: str
) -> None:
    source = """
        qfunc main(output res: qbit[]) {
            x: qbit[];
            lsb: qbit;
            msb: qnum<2>;
            allocate(3, x);
            x -> {lsb, msb};
    """ + transfer + "}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
