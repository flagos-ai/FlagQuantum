"""Check original Hadamard tests with a hidden, entangled QFT target."""

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


@pytest.mark.parametrize("name", ["hadamard_test.qmod", "hadamard_test_tutorial.qmod"])
def test_original_hadamard_test_full_joint_state(name: str) -> None:
    path = FIXTURES / name
    raw = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "1759331eb973dc306eed7a9e17c25efb9cf4b61f"
    )
    axis = torch.arange(16, dtype=torch.float64)
    qft = torch.exp((2j * math.pi / 16) * torch.outer(axis, axis)) / 4
    identity = torch.eye(16, dtype=torch.complex128)
    reference = fq.Circuit(5, dtype=torch.complex128).h(0)
    reference.any(*range(5), matrix=torch.block_diag(identity, qft)).h(0)
    program = synth.synthesize(path, max_workspace=0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {"expectation_value": (0,)}
    assert dict(
        program.probabilities(result, registers=("expectation_value",))
    ) == pytest.approx({(0,): 5 / 8, (1,): 3 / 8})
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_qubits == 5
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(path, max_qubits=4)


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(x); drop(x);", "local qubit array"),
        ("allocate(x); y:qbit[2]; drop(y);", "initialized hidden storage"),
        ("allocate(x); y:qbit[2]; allocate(y);", "drop every hidden register"),
        (
            "allocate(x); y:qbit[2]; allocate(y); drop(y); H(y[0]);",
            "cannot be used again",
        ),
    ],
)
def test_drop_checks_lifetime_and_ownership(body: str, diagnostic: str) -> None:
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(f"qfunc main(output x:qbit) {{{body}}}", source_format="qmod")


def test_reject_recursive_writable_helper():
    source = """qfunc loop(a:qbit) {within {H(a);} apply {loop(a);}}
    qfunc main(output a:qbit) {allocate(a); loop(a);}"""
    with pytest.raises(InteropConversionError, match="acyclic"):
        synth.synthesize(source, source_format="qmod")
