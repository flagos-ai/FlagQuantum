"""Run the native Python counterpart of the pinned SWAP.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import SWAP, Output, QBit, allocate, qfunc


@qfunc
def main(q1: Output[QBit], q2: Output[QBit]) -> None:
    allocate(q1)
    allocate(q2)
    SWAP(q1, q2)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(
        dict(
            program.probabilities(
                result, registers=("q1", "q2"), max_amplitudes=1 << 24
            )
        )
    )
