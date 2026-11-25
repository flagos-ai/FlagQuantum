"""Native high-level counterpart of the exact hadamard_transform.qmod original."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(x: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
