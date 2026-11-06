"""Check Qmod controlled phase flips and their interference."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/Qmod_tutorial_part1.qmod"
)
ALL = "control(x) {phase(pi);}"
ELEMENT = "control(x[3]) {phase(pi);}"


@pytest.mark.parametrize("case", ["original", "array", "element"])
def test_controlled_phase_full_state(case: str) -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5507361161a5e78ebd4663dd08c7191d70b24d00"
    )
    if case == "original":
        program = synth.synthesize(SOURCE)
    else:
        control = ALL if case == "array" else ELEMENT
        program = synth.synthesize(
            f"""qfunc main(output x:qbit[]) {{
                allocate(4,x); hadamard_transform(x); {control}
            }}""",
            source_format="qmod",
        )
    diagonal = torch.ones(16, dtype=torch.complex128)
    if case in {"original", "array"}:
        diagonal[15] *= -1
    if case in {"original", "element"}:
        diagonal[8:] *= -1
    assert diagonal[15] == (1 if case == "original" else -1)

    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.any(*range(4), matrix=torch.diag(diagonal))
    if case == "original":
        for wire in range(4):
            reference.h(wire)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == (0 if case == "element" else 1)


@pytest.mark.parametrize("condition", ["x", "x[0]"])
def test_single_element_phase(condition: str) -> None:
    program = synth.synthesize(
        f"""qfunc main(output x:qbit[]) {{
            allocate(1,x); hadamard_transform(x); control({condition}) {{phase(pi);}}
        }}""",
        source_format="qmod",
    )
    expected = torch.tensor([1, -1], dtype=torch.complex128) / 2**0.5
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


def test_array_phase_honors_workspace_budget() -> None:
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(SOURCE, max_workspace=0)
