"""Pauli Z evolution as native qfunc phase intent, Qmod and an independent Circuit.

Run this module for the small three-form example. Reproduce the unchanged
12-bit originals with tools/validate_qmod_large_pauli.py; this example does not
claim a native Python qfunc Hamiltonian-array API."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, allocate, hadamard_transform, phase, qfunc


@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    hadamard_transform(q)
    # exp(-i*0.37*Z): the zero and one amplitudes acquire opposite phases.
    phase(1, -0.37)
    phase(q, 0.74)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
