"""Run the native Python counterpart of the pinned R.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, R, allocate, qfunc


@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    R(1, 2, q)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("q",), max_amplitudes=1 << 24)))
