"""Compare the pinned original, native qfunc and independent Circuit."""

import math
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.functions.state_preparation.prepare_uniform_interval_state.main import (
    main,
)
from flagquantum import synth
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state


def run_example() -> None:
    expected = torch.zeros(1 << 5, dtype=torch.complex128)
    expected[2:8] = 1 / math.sqrt(6)
    circuit = fq.Circuit(5, dtype=torch.complex128)
    append_arbitrary_state(circuit, expected, range(5))
    reference = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(reference, expected, atol=1e-7, rtol=0)
    for program in (
        synth.synthesize(main),
        synth.synthesize(Path(__file__).with_name("source.qmod"), source_format="qmod"),
    ):
        actual = fq.run(program).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=0)
        print(program.report["resources"])
    print("Original Qmod, native qfunc and independent Circuit agree.")


if __name__ == "__main__":
    run_example()
