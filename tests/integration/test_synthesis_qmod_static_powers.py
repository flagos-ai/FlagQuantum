"""Check exact static expansion and a phase-kickback local against Circuits."""

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


def bv_reference() -> torch.Tensor:
    circuit = fq.Circuit(6, dtype=DTYPE)
    for wire in range(5):
        circuit.h(wire)
    circuit.x(5).h(5)
    for wire in (4, 2, 1):
        circuit.cx(wire, 5)
    circuit.h(5).x(5)
    for wire in range(5):
        circuit.h(wire)
    return fq.run(circuit).statevector().reshape(32, 2)


def test_original_bernstein_vazirani_matches_clean_six_qubit_circuit() -> None:
    manifest = json.loads((FIXTURES / "static_powers_sources.json").read_text())
    original = manifest["fixtures"][0]
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

    program = synth.synthesize(path, max_qubits=5, max_workspace=0)
    result = fq.run(program)
    reference = bv_reference()
    actual = torch.zeros_like(reference)
    actual[:, 0] = result.statevector().reshape(32)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
    torch.testing.assert_close(reference[:, 1], torch.zeros_like(reference[:, 1]))
    assert dict(program.probabilities(result, registers=("x",))) == pytest.approx(
        {(13,): 1.0}
    )
    assert program.report["workspace_qubits"] == 0


def test_nested_static_loops_and_exact_powers_in_angles_and_bounds() -> None:
    source = """
        W:int = 2 ** 2;
        qfunc rotate(output_bit:qbit) {
          repeat (i: W - 2) {
            repeat (j: 2 ** i) { RZ(pi / (2 ** (j + 2)), output_bit); }
          }
        }
        qfunc main(output x:qbit) {allocate(x); H(x); rotate(x);}
    """
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(1, dtype=DTYPE).h(0).rz(0, math.pi * 5 / 8)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("repeat (i: 2 ** 256) {}", "exponent must be in 0..255"),
        ("repeat (i: 4 ** 128) {}", "static integer exceeds 256 bits"),
        ("repeat (i: 2 ** -1) {}", "exponent must be in 0..255"),
        ("repeat (i: 4 % 0) {}", "positive divisor"),
        ("repeat (i: 256) { repeat (j: 256) {} }", "loops exceed 4096 iterations"),
        ("repeat (i: 2) { repeat (i: 2) {} }", "loop index cannot shadow"),
        ("RZ(2 ** x, x);", "expected a static integer"),
        ("free(x);", "requires a live local qbit"),
        (
            "a:qbit; allocate(a); X(a); free(a);",
            "cannot prove local qbit returned to zero",
        ),
        (
            "a:qbit; allocate(a); X(a); H(a); CX(a,x); H(a); X(a); free(a);",
            "cannot prove local qbit returned to zero",
        ),
        (
            "a:qbit; allocate(a); X(a); H(a); CX(x,a); H(a); X(a); free(a); H(a);",
            "cannot be used again",
        ),
    ],
)
def test_static_expansion_and_free_reject_unproven_effects(
    body: str, diagnostic: str
) -> None:
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            "qfunc main(output x:qbit) {allocate(x);" + body + "}",
            source_format="qmod",
        )


def test_kickback_preserves_coherent_phases() -> None:
    source = """
        qperm kick(const q:qbit[], t:qbit) {CX(q[0],t);}
        qfunc flag(q:qbit[2]) {
            a:qbit;
            allocate(a);
            within {X(a);H(a);} apply {kick(q,a);}
            free(a);
        }
        qfunc main(output x:qbit[2]) {
            allocate(x); hadamard_transform(x); RZ(0.27,x[1]); flag(x);
        }
    """
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1).rz(0, 0.27).z(1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
