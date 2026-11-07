"""Compare Qmod and native qfunc against the independent Circuit."""

import math
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.main import (
    one,
    two,
    zero,
)
from flagquantum import synth
from flagquantum.services.synthesis import _synthesize

QMOD = Path(__file__).with_name("source.qmod").read_text(encoding="utf-8")


def run_example() -> None:
    # Physical order is flag, data; A applies H(data) followed by CX(data,flag).
    a = torch.tensor(
        [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]],
        dtype=torch.complex128,
    ) / math.sqrt(2)
    oracle = torch.diag(torch.tensor([1, 1, -1, -1], dtype=torch.complex128))
    reflection = torch.diag(torch.tensor([1, -1, -1, -1], dtype=torch.complex128))
    grover = a @ reflection @ a.conj().T @ oracle
    for k, function in enumerate((zero, one, two)):
        circuit = fq.Circuit(2, dtype=torch.complex128).any(0, 1, matrix=a)
        for _ in range(k):
            circuit.any(0, 1, matrix=grover)
        expected = fq.run(circuit).statevector().reshape(-1)
        for p in (
            synth.synthesize(function),
            _synthesize(QMOD, source_format="qmod", classical_inputs={"k": k}),
        ):
            result = fq.run(p)
            state = result.statevector().reshape(4, -1)
            torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
            assert float(state[:, 1:].abs().square().sum()) < 1e-24
            print(k, dict(p.probabilities(result, registers=("flag",))))
            print(p.report["resources"])
    print(
        "Native scalar phase composition, Qmod and Circuit agree, including absolute round signs."
    )


if __name__ == "__main__":
    run_example()
