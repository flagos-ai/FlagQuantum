"""Compare Qmod and native qfunc against the independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.hidden_shift.main import (
    main,
)
from flagquantum import synth


def run_example() -> None:
    source = Path(__file__).with_name("source.qmod")
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
    circuit.x(3).cz(3, 2).x(3).cz(1, 0)
    for bit in range(4):
        circuit.h(bit)
    circuit.cz(3, 2).cz(1, 0)
    for bit in range(4):
        circuit.h(bit)
    expected = fq.run(circuit).statevector().reshape(-1)
    for program in (synth.synthesize(source), synth.synthesize(main)):
        actual = fq.run(program).statevector().reshape(16, -1)
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-24
        print(program.report["resources"])
    print(
        "Original Qmod, native qfunc and Circuit agree: +|0001>, with clean workspace."
    )


if __name__ == "__main__":
    run_example()
