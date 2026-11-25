"""Independently verify complete native QPE and adaptively visited Grover states."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_qpe.comparisons.circuit import (
    reference_circuit as qpe_reference,
)

from ..main import run_iterative, sample_qpe


def run_example() -> None:
    phase = sample_qpe()
    torch.testing.assert_close(
        fq.run(phase.program).statevector(),
        fq.run(qpe_reference()).statevector(),
        atol=1e-11,
        rtol=0,
    )
    _, sampler = run_iterative(1 / (16 * 15))
    for k, program in sampler.programs.items():
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(k)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        assert program.report["workspace_qubits"] == 0
        print(
            "Grover power:",
            k,
            "maximum absolute state error:",
            float((actual - expected).abs().max()),
        )
    print(
        "QPE and all visited Grover fullstates agree, including absolute phase and retained data."
    )


if __name__ == "__main__":
    run_example()
