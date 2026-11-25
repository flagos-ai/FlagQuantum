"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.large_pauli.main import main
from flagquantum import synth


def run_example() -> None:
    reference = (
        fq.run(fq.Circuit(1, dtype=torch.complex128).h(0).rz(0, theta=0.74))
        .statevector()
        .reshape(-1)
    )
    for program in (synth.synthesize(main),):
        actual = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 1
        expected = torch.zeros_like(actual)
        expected[:: 1 << workspace] = reference
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
