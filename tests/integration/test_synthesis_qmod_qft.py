"""Verify the pinned Qmod QFT and the full Fourier matrix convention."""

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/qft.qmod"
WIDTH = 4
SIZE = 1 << WIDTH


def fourier_matrix() -> torch.Tensor:
    indices = torch.arange(SIZE, dtype=torch.float64)
    return torch.exp((2j * math.pi / SIZE) * torch.outer(indices, indices)) / math.sqrt(
        SIZE
    )


def test_original_qft_full_state_and_resources() -> None:
    source = FIXTURE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        == "fc13d0dd2cd71754d6bf7e63b4a654753b4e33a9"
    )
    program = synth.synthesize(FIXTURE)
    reference = fq.Circuit(WIDTH, dtype=torch.complex128)
    reference.any(*range(WIDTH), matrix=fourier_matrix())
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    resources = program.report["resources"]
    assert resources.n_qubits == WIDTH
    assert resources.operation_counts == {"h": 4, "cphase": 6, "swap": 2}
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("basis", range(SIZE))
def test_qft_columns_and_offset_wires(basis: int) -> None:
    preparation = "".join(
        f"X(x[{i}]);" for i, bit in enumerate(reversed(f"{basis:04b}")) if bit == "1"
    )
    program = synth.synthesize(
        f"""qfunc main(output sentinel:qbit, output x:qbit[]) {{
            allocate(sentinel); X(sentinel); allocate(4,x); {preparation} qft(x);
        }}""",
        source_format="qmod",
    )
    reference = fq.Circuit(WIDTH + 1, dtype=torch.complex128)
    reference.x(0)
    for i, bit in enumerate(f"{basis:04b}", start=1):
        if bit == "1":
            reference.x(i)
    reference.any(*range(1, WIDTH + 1), matrix=fourier_matrix())
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        ("qfunc main(output x:qbit[]) {qft(x);}", "needs a declared width"),
        ("qfunc main(output x:qbit) {allocate(x); qft(x);}", "requires a qubit array"),
        (
            "qfunc main(output x:qbit[]) {allocate(4,x); qft(x[0]);}",
            "one output variable",
        ),
    ],
)
def test_qft_refusals(source: str, diagnostic: str) -> None:
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
