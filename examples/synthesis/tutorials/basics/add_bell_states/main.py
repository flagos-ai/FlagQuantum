"""Add two independently prepared Bell registers without designing an adder."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_state, qfunc


@qfunc
def main(a: Output[QNum], b: Output[QNum], total: Output[QNum]) -> None:
    prepare_state([1 / 2, 0, 0, 1 / 2], 0, a)
    prepare_state([0, 1 / 2, 1 / 2, 0], 0, b)
    total |= a + b


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "total"))))
