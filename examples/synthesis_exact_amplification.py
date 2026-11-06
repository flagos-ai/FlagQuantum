"""Exact amplification original, existing native composition and independent Circuit.

The Qmod entry owns its ancilla. The native comparison spells the same small
algorithm through existing qfunc phase/rotation composition and returns its
explicit clean auxiliary register; it adds no native exact-amplification API.
"""

import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, R, allocate, phase, qfunc

PHI = math.pi / 2
THETA = 2 * math.asin(math.sqrt(0.07))
BETA = 2 * math.acos(math.sin(math.pi / 14) / math.sqrt(0.07))


@qfunc
def main(target: Output[QBit], aux: Output[QBit]) -> None:
    allocate(target)
    allocate(aux)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    # The current native frontend uses static calls; three rounds are explicit.
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)
    phase(target, math.pi)
    phase(target * aux, -math.pi)
    R(-BETA, PHI, aux)
    R(-THETA, PHI, target)
    phase(target, -math.pi)
    phase(aux, -math.pi)
    phase(target * aux, math.pi)
    R(THETA, PHI, target)
    R(BETA, PHI, aux)


def run_example() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/classiq_qmod/exact_amplitude_amplification.qmod"
    )
    c = fq.Circuit(2, dtype=torch.complex128).ry(0, theta=THETA).ry(1, theta=BETA)
    for _ in range(3):
        c.x(1).cz(0, 1).x(1)
        c.ry(1, theta=-BETA).ry(0, theta=-THETA)
        c.z(0).z(1).cz(0, 1)
        c.ry(0, theta=THETA).ry(1, theta=BETA)
    reference = fq.run(c).statevector().reshape(-1)
    for program in (synth.synthesize(source), synth.synthesize(main)):
        state = fq.run(program).statevector().reshape(4, -1)
        torch.testing.assert_close(state[:, 0], reference, atol=1e-12, rtol=0)
        assert float(state[:, 1:].abs().square().sum()) < 1e-24
        print(program.report["resources"])
    print(
        "Original Qmod, native phase composition and Circuit agree: +|10>; auxiliary zero."
    )


if __name__ == "__main__":
    run_example()
