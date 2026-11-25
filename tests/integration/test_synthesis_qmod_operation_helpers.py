"""Validate reuse of Pauli evolution and unitary lowering in Qmod helpers."""

import ast
import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def _original(name: str, blob: str) -> Path:
    path = FIXTURES / name
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    return path


@pytest.mark.parametrize("theta", [0.23, -0.41])
def test_original_adapt_vqe(theta):
    source = _original("adapt_vqe.qmod", "3bd4550d1ddcca756681abcc76758cd8b997a662")
    program = _synthesize(source, classical_inputs={"params": [theta]})
    reference = fq.Circuit(4, dtype=DTYPE).rxx(0, 3, theta=2 * theta)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("prepare", [False, True])
def test_quantum_volume_matrix_order_repetition_and_spectator(prepare):
    source = _original(
        "quantum_volume.qmod", "e82274ef0ac4cff694e6c51746c94c59844b32ee"
    )
    matrix_literal = source.read_text().split("unitary(", 1)[1].split(", {target", 1)[0]
    matrix = torch.tensor(ast.literal_eval(matrix_literal), dtype=DTYPE)
    reference = fq.Circuit(3, dtype=DTYPE)
    if prepare:
        program = synth.synthesize(
            source.read_text().replace(
                "allocate(3, target);",
                "allocate(3, target); H(target[0]); X(target[1]);",
            ),
            source_format="qmod",
        )
        reference.h(2).x(1)
    else:
        program = synth.synthesize(source)
    for _ in range(3):
        reference.any(0, 1, matrix=matrix)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_indexed_unitary_can_span_distinct_registers():
    source = """qfunc main(output a:qbit, output b:qbit) {
        allocate(a); allocate(b); X(a);
        unitary([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], {b,a});
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).x(0).cx(0, 1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("target", "setup", "diagnostic"),
    [
        ("{a,a}", "allocate(a); allocate(b);", "distinct qubit locations"),
        ("{a,b}", "allocate(a);", "unitary requires initialized qubits"),
    ],
)
def test_indexed_unitary_checks_alias_and_initialization(target, setup, diagnostic):
    source = (
        f"qfunc main(output a:qbit,output b:qbit) {{ {setup}"
        f"unitary([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], {target}); }}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_repeat_bounds_even_empty_helper_expansion():
    source = """
      qfunc empty(q:qbit) {}
      qfunc layer(q:qbit) { repeat (j:256) { empty(q); } }
      qfunc main(output q:qbit) { allocate(q); repeat (i:256) { layer(q); } }
    """
    with pytest.raises(
        InteropConversionError, match="static loops exceed 4096 iterations"
    ):
        synth.synthesize(source, source_format="qmod")
