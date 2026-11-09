"""Verify bounded Qmod unitary literals and exact complex phases."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/unitary.qmod"
DTYPE = torch.complex128


@pytest.mark.parametrize("basis", [0, 1, 2, 3, None])
def test_original_unitary_and_phase_sensitive_inputs(basis: int | None) -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5ddd508e6676254ab743256583b1dc2b7e67670a"
    )
    reference = fq.Circuit(2, dtype=DTYPE)
    if basis == 0:
        program = synth.synthesize(SOURCE)
    else:
        if basis is None:
            preparation = "hadamard_transform(x);"
            reference.h(0).h(1)
        else:
            preparation = "".join(f"X(x[{i}]);" for i in range(2) if basis & (1 << i))
            for wire in range(2):
                if basis & (2 >> wire):
                    reference.x(wire)
        program = synth.synthesize(
            raw.decode().replace("unitary(", preparation + "unitary("),
            source_format="qmod",
        )
    reference.any(0, 1, matrix=torch.diag(torch.tensor([1, 1, -1j, 1j], dtype=DTYPE)))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_non_diagonal_matrix_wire_order():
    source = """qfunc main(output sentinel:qbit, output x:qbit[]) {
        allocate(sentinel); X(sentinel); allocate(2,x); X(x[1]);
        unitary([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],x);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(3, dtype=DTYPE).x(0).x(1).cx(1, 2)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_single_qubit_preserves_global_phase():
    source = """qfunc main(output x:qbit[]) {
        allocate(1,x); unitary([[0,1j],[1j,0]],x);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        torch.tensor([0, 1j], dtype=DTYPE),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("matrix", "diagnostic"),
    [
        ("[[1,0],[0,2]]", "must be unitary"),
        ("[[1,0],[1]]", "square matrix"),
        ("[[1,0],[0,1 j]]", "expected"),
    ],
)
def test_reject_invalid_matrices(matrix, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output x:qbit[]) {{allocate(1,x); unitary({matrix},x);}}",
            source_format="qmod",
        )
