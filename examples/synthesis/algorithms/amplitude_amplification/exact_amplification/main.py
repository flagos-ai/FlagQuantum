"""Exact amplification original, existing native composition and independent Circuit.

The Qmod entry owns its ancilla. The native comparison spells the same small
algorithm through existing qfunc phase/rotation composition and returns its
explicit clean auxiliary register; it adds no native exact-amplification API."""

import math

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
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
