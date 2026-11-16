"""Native high-level counterpart of the exact classical_shadows.qmod original."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_bell_state, qfunc


@qfunc
def main(qarr: Output[QNum]) -> None:
    prepare_bell_state(1, qarr)


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("qarr",))))
