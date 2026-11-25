"""Test-only unchanged source and independent molecular Circuit acceptance."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from tests.integration.test_synthesis_native_molecular_walk import ORIGINAL


def run_example() -> None:
    program = synth.synthesize(
        ORIGINAL, max_qubits=15, max_workspace=2, max_operations=60_000
    )
    actual = fq.run(program).statevector().reshape(8192, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
