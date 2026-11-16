"""Run the native Python counterpart of the pinned prepare_bell_state.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_bell_state, qfunc


@qfunc
def main(x: Output[QNum]) -> None:
    prepare_bell_state(2, x)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",), max_amplitudes=1 << 24)))
