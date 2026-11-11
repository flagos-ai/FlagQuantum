"""Complete native Vlasov-Ampere acceptance against independent Circuits."""

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.plasma.vlasov_ampere.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.plasma.vlasov_ampere.main import build_program
from flagquantum.compiler import ResourceEstimate
from tests.integration.test_synthesis_qmod_vlasov import original, vlasov_reference

pytestmark = pytest.mark.integration


def test_native_vlasov_full_state_absolute_phase_and_clean_workspace() -> None:
    source = original()
    program = build_program()
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 24
    assert program.report["workspace_qubits"] == 2
    assert resources.n_operations == 22976
    assert resources.depth == 18578
    assert tuple(program.registers) == (
        "electric",
        "x",
        "velocity",
        "reference_e",
        "reference_x",
        "reference_v",
        "selector",
        "second",
        "first",
        "indicator",
        "block",
        "zeta",
        "flag",
    )
    # Both references construct mathematical local matrices independently of synth.
    # The source-derived one also detects accidental changes to frozen constants.
    expected = fq.run(reference_circuit()).statevector().flatten()
    frozen_expected = (
        fq.run(vlasov_reference(source.read_text())).statevector().flatten()
    )
    actual = fq.run(program).statevector().reshape(1 << 22, 4)
    dirty = 0.0
    for start in range(0, len(expected), 65536):
        stop = start + 65536
        torch.testing.assert_close(
            expected[start:stop], frozen_expected[start:stop], atol=1e-13, rtol=0
        )
        torch.testing.assert_close(
            actual[start:stop, 0], expected[start:stop], atol=1e-11, rtol=0
        )
        dirty += float(actual[start:stop, 1:].abs().square().sum())
    assert dirty < 1e-22
