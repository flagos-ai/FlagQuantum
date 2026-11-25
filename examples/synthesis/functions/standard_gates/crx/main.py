"""Run the native Python counterpart of the pinned CRX.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CRX, Output, QBit, allocate, qfunc


@qfunc
def main(q_target: Output[QBit], q_control: Output[QBit]) -> None:
    allocate(q_target)
    allocate(q_control)
    CRX(1, q_control, q_target)


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(
        dict(
            program.probabilities(
                result, registers=("q_target", "q_control"), max_amplitudes=1 << 24
            )
        )
    )
