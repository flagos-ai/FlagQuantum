"""One-bit swap test in native qfunc, Qmod and public Circuit syntax.

The pinned three-bit original has non-normalized vectors and remains refused.
Run tools/validate_qmod_swap_test.py for its labelled diagnostic comparison.
This example uses valid one-bit states and existing native phase/CX primitives;
it adds no native array preparation or swap_test export.
"""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QBit,
    R,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

THETA = 0.74
PHI = -0.39
HALF_PI = math.pi / 2
PI = math.pi


@qfunc
def main(test: Output[QBit], first: Output[QBit], second: Output[QBit]) -> None:
    allocate(test)
    allocate(first)
    allocate(second)
    R(THETA, HALF_PI, first)
    R(PHI, HALF_PI, second)
    hadamard_transform(test)
    CX(second, first)
    hadamard_transform(second)
    phase(test * first * second, PI)
    hadamard_transform(second)
    CX(second, first)
    hadamard_transform(test)


QMOD = """qfunc main(output test:qbit,output first:qbit[1],output second:qbit[1]) {
 allocate(first);allocate(second);RY(0.74,first[0]);RY(-0.39,second[0]);
 swap_test(first,second,test);
}"""


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
