"""Check the pinned flexible QPE programs against independent Circuits."""

import hashlib
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


@pytest.mark.parametrize("name", ["qpe.qmod", "qpe_flexible.qmod"])
def test_original_qpe_full_state_and_phase_readout(name: str) -> None:
    path = FIXTURES / name
    raw = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "d93c2d144b8c6d2f67abe7de4809b566b9ba6a88"
    )
    reference = fq.Circuit(4, dtype=DTYPE).h(0).h(1).cx(0, 3).cx(1, 2)
    program = synth.synthesize(path)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0
    scale = 1 << program.report["qmod_fractional_bits"]["phase"]
    decoded = {
        (state, phase / scale): probability
        for (state, phase), probability in program.probabilities(
            result, registers=("state", "phase")
        ).items()
    }
    assert decoded == pytest.approx(
        {(0, 0): 0.25, (1, 0.5): 0.25, (2, 0.25): 0.25, (3, 0.75): 0.25}
    )


def test_controlled_non_diagonal_pauli_and_inverse_fourier():
    source = """qfunc main(output x:qbit[], output p:qnum<2,UNSIGNED,2>) {
        allocate(1,x); allocate(p);
        qpe_flexible(lambda(k) {
            suzuki_trotter([PauliTerm {pauli=[Pauli::Y], coefficient=0.17}],
                           (-2*pi)*k, 1, 1, x);
        }, p);
    }"""
    reference = fq.Circuit(3, dtype=DTYPE).h(1).h(2)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    identity = torch.eye(2, dtype=DTYPE)
    for wire, power in ((2, 1), (1, 2)):
        angle = 2 * math.pi * 0.17 * power
        rotation = math.cos(angle) * identity + 1j * math.sin(angle) * y
        reference.any(wire, 0, matrix=torch.block_diag(identity, rotation))
    axis = torch.arange(4, dtype=torch.float64)
    reference.any(1, 2, matrix=torch.exp(-0.5j * math.pi * torch.outer(axis, axis)) / 2)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("allocate(phase);", "", "initialized by allocate"),
        ("qnum<2, UNSIGNED, 2>", "qnum<2>", "all-fractional phase output"),
        ("}, phase);", "}, phase); H(phase);", "terminal output"),
        ("power, 1, 1", "power, 1, 2", "one repetition"),
        ("lambda(power)", "lambda(state)", "unshadowed"),
    ],
)
def test_qpe_rejects_out_of_profile_calls(old, new, diagnostic):
    source = (FIXTURES / "qpe.qmod").read_text().replace(old, new, 1)
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
