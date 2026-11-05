"""Compare the pinned exponential state with independent matrix Circuits."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/prepare_exponential_state.qmod"
)


def test_original_exponential_state_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "8ebfc5c90d89de0f293538f2d1a1afe211d7ea75"
    )
    probabilities = torch.softmax(-0.1 * torch.arange(32, dtype=torch.float64), dim=0)
    identity = torch.eye(32, dtype=torch.complex128)
    delta = identity[:, 0] - probabilities.sqrt()
    reference = fq.Circuit(5, dtype=torch.complex128)
    reference.any(
        *range(5),
        matrix=identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square(),
    )
    program = synth.synthesize(SOURCE)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].operation_counts == {"ry": 5}


@pytest.mark.parametrize("rate", [-0.4, 0.0, 0.23, -float(2**53), float(2**53)])
def test_exponential_state_rotates_existing_input(rate: float) -> None:
    program = synth.synthesize(
        """qfunc main(output q:qbit[], rate:real) {
            allocate(2,q); X(q[0]); prepare_exponential_state(rate,q);
        }""",
        source_format="qmod",
        classical_inputs={"rate": rate},
    )
    reference = fq.Circuit(2, dtype=torch.complex128)
    reference.x(1)
    for wire, weight in enumerate((2, 1)):
        p0, p1 = torch.softmax(
            torch.tensor(
                [min(rate, 0.0) * weight, -max(rate, 0.0) * weight], dtype=torch.float64
            ),
            dim=0,
        ).sqrt()
        reference.any(
            wire, matrix=torch.stack((torch.stack((p0, -p1)), torch.stack((p1, p0))))
        )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_exponential_state_requires_initialized_storage() -> None:
    with pytest.raises(InteropConversionError, match="needs a declared width"):
        synth.synthesize(
            "qfunc main(output q:qbit[]) {prepare_exponential_state(0.1,q);}",
            source_format="qmod",
        )
