"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.arithmetic.boolean_factors.main import main
from flagquantum.synth import synthesize


def circuit_reference() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    diagonal = torch.ones(16, dtype=torch.complex128)
    diagonal[-1] = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
    circuit.any(0, 1, 2, 3, matrix=torch.diag(diagonal))
    return circuit


def run_example() -> None:
    expected = fq.run(circuit_reference()).statevector().reshape(-1)
    for source in (main,):
        program = synthesize(source, memory_limit_bytes=64 * 1024**2)
        actual = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 4
        embedded = torch.zeros_like(actual)
        embedded[:: 1 << workspace] = expected
        torch.testing.assert_close(actual, embedded, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
