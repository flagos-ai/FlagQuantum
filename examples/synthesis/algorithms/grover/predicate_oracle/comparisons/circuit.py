"""Compare native FlagQuantum and independent Circuit amplitudes."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.grover.predicate_oracle.main import main
from flagquantum import synth


def run_example() -> None:
    diffuser = torch.full((4, 4), 0.5, dtype=torch.complex128) - torch.eye(
        4, dtype=torch.complex128
    )
    circuit = (
        fq.Circuit(2, dtype=torch.complex128).h(0).h(1).cphase(0, 1, theta=math.pi)
    )
    circuit.any(0, 1, matrix=diffuser)
    reference = fq.run(circuit).statevector().reshape(-1)
    for program in (synth.synthesize(main),):
        actual = fq.run(program).statevector().reshape(-1)
        expected = torch.zeros_like(actual)
        expected[:: 1 << program.circuit.n_wires - 2] = reference
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
