"""Run the native Python counterpart of the pinned example_prepare_state.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_state, qfunc


@qfunc
def main(x: Output[QNum]) -> None:
    prepare_state([0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06], 0.01, x)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",), max_amplitudes=1 << 24)))
