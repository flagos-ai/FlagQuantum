"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main import main
from examples.synthesis.tutorials.composition.python_program.comparisons.circuit import (
    circuit_reference,
)
from flagquantum import synth


def run() -> None:
    program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
    result = fq.run(program)
    outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert outcomes.keys() == expected.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for (key, value) in expected.items())
    state = result.statevector().reshape(4, 4, 16, -1)
    reference = fq.run(circuit_reference()).statevector().reshape(4, 4, 16, 2)
    torch.testing.assert_close(state[..., 0], reference[..., 0], atol=1e-12, rtol=0)
    assert float(state[..., 1:].abs().square().sum()) < 1e-24
    print(outcomes)
    print(program.report["resources"])
    print("Workspace:", program.report["workspace_qubits"])


if __name__ == "__main__":
    run()
