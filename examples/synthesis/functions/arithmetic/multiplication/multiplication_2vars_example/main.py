"""Run the native Python counterpart of the pinned multiplication_2vars_example.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, qfunc


@qfunc
def main(a: Output[QNum], b: Output[QNum], res: Output[QNum]) -> None:
    a |= 4
    b |= 5
    res |= a * b


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(
        dict(
            program.probabilities(
                result, registers=("a", "b", "res"), max_amplitudes=1 << 24
            )
        )
    )
