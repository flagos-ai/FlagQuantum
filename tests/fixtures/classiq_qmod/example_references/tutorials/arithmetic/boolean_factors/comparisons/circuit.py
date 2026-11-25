"""Compare Qmod and native qfunc against the independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.arithmetic.boolean_factors.main import main
from flagquantum.synth import (
    synthesize,
)


def circuit_reference() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    diagonal = torch.ones(16, dtype=torch.complex128)
    diagonal[-1] = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
    circuit.any(0, 1, 2, 3, matrix=torch.diag(diagonal))
    return circuit


QMOD = Path(__file__).with_name("source.qmod").read_text(encoding="utf-8")


def run_example() -> None:
    expected = fq.run(circuit_reference()).statevector().reshape(-1)
    for source in (main, QMOD):
        program = (
            synthesize(source, source_format="qmod", memory_limit_bytes=64 * 1024**2)
            if isinstance(source, str)
            else synthesize(source, memory_limit_bytes=64 * 1024**2)
        )
        actual = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 4
        embedded = torch.zeros_like(actual)
        embedded[:: 1 << workspace] = expected
        torch.testing.assert_close(actual, embedded, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print("Native qfunc and Qmod match Circuit, including clean workspace.")


if __name__ == "__main__":
    run_example()
