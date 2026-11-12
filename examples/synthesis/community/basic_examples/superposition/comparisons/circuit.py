"""Verify the entire local preparation and visualization tutorial."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.basic_examples.superposition.main import main
from examples.synthesis.compare_states import compare_states
from flagquantum.drawer import draw
from flagquantum.synth import synthesize


def run_example() -> None:
    program = synthesize(main)
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)
    compare_states(program, reference, ("register_a",))
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(),
        torch.full((8,), 1 / math.sqrt(8), dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    outcomes = dict(program.probabilities(result, registers=("register_a",)))
    assert outcomes.keys() == {(word,) for word in range(8)}
    assert all(abs(probability - 1 / 8) < 1e-12 for probability in outcomes.values())
    assert program.report["workspace_qubits"] == 0
    diagram = draw(program.circuit)
    assert isinstance(diagram, str) and diagram.strip()
    print(diagram)
    print(outcomes)


if __name__ == "__main__":
    run_example()
