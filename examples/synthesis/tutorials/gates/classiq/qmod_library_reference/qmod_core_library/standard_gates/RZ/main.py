"""Run the native Python counterpart of the pinned RZ.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import RZ, Output, QBit, allocate, qfunc


@qfunc
def main(q: Output[QBit]) -> None:
    allocate(q)
    RZ(1.9, q)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("q",), max_amplitudes=1 << 24)))
