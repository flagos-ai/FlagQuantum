"""Large native composition executes through the ordinary public program path."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


def test_expanded_native_effects_preserve_phase_and_complete_public_execution() -> None:
    entry = program_definition(
        "from flagquantum.synth import RX, phase\n"
        "@qfunc\ndef main(q:Output[QNum[1]]):\n"
        "    allocate(q)\n"
        "    for i in range(240):\n"
        "        for j in range(16):\n"
        "            RX(0.001,q)\n"
        "            RX(0.001,q)\n"
        "    phase(1,0.31)\n"
    )
    program = synth.synthesize(
        entry,
        max_qubits=1,
        max_workspace=0,
        max_operations=8000,
        memory_limit_bytes=1024**2,
    )
    state = fq.run(program).statevector().flatten()
    expected = torch.tensor(
        [math.cos(7.68 / 2), -1j * math.sin(7.68 / 2)], dtype=torch.complex128
    ) * complex(math.cos(0.31), math.sin(0.31))
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert program.report["source_format"] == "python"
    assert program.report["workspace_qubits"] == 0
    assert dict(
        program.probabilities(fq.run(program), registers=("q",))
    ) == pytest.approx(
        {
            (0,): float(expected[0].abs().square()),
            (1,): float(expected[1].abs().square()),
        }
    )
