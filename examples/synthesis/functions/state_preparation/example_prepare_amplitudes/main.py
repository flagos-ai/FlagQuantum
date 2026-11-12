"""Run the native Python counterpart of the pinned example_prepare_amplitudes.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_amplitudes, qfunc


@qfunc
def main(x: Output[QNum]) -> None:
    prepare_amplitudes(
        [
            -0.540061724867322,
            -0.38575837490523,
            -0.231455024943138,
            -0.077151674981046,
            0.077151674981046,
            0.231455024943138,
            0.38575837490523,
            0.540061724867322,
        ],
        0,
        x,
    )


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",), max_amplitudes=1 << 24)))
