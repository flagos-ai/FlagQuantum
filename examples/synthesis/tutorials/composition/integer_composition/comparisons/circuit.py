"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.composition.integer_composition.main import main
from flagquantum import synth


def run_example() -> None:
    reference = (
        fq.run(fq.Circuit(2, dtype=torch.complex128).h(0).h(1).cphase(0, 1, theta=1.2))
        .statevector()
        .reshape(-1)
    )
    programs = (synth.synthesize(main),)
    for program in programs:
        state = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 2
        expected = torch.zeros_like(state)
        expected[:: 1 << workspace] = reference
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
