"""Compare Qmod and native qfunc against the independent Circuit."""

import math
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.state_preparation.swap_test.main import (
    PHI,
    THETA,
    main,
)
from flagquantum import synth

QMOD = Path(__file__).with_name("source.qmod").read_text(encoding="utf-8")


def run_example() -> None:
    circuit = fq.Circuit(3, dtype=torch.complex128).ry(1, THETA).ry(2, PHI).h(0)
    swap = torch.eye(8, dtype=torch.complex128)
    swap[5, 5] = swap[6, 6] = 0
    swap[5, 6] = swap[6, 5] = 1
    circuit.any(0, 1, 2, matrix=swap).h(0)
    expected = fq.run(circuit).statevector().reshape(-1)
    probability = (1 + math.cos((THETA - PHI) / 2) ** 2) / 2
    for program in (
        synth.synthesize(main),
        synth.synthesize(QMOD, source_format="qmod"),
    ):
        result = fq.run(program)
        state = result.statevector().reshape(8, -1)
        torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
        assert float(state[:, 1:].abs().square().sum()) < 1e-24
        observed = dict(program.probabilities(result, registers=("test",)))
        assert abs(observed[(0,)] - probability) < 1e-12
        print(observed)
        print(program.report["resources"])
    print("Native phase composition, Qmod swap_test and independent Circuit agree.")


if __name__ == "__main__":
    run_example()
