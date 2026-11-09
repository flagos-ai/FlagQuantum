"""Run the native Python counterpart of the pinned synthesis_tutorial.qmod."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    y |= x**2 + 1


if __name__ == "__main__":
    program = synth.synthesize(main, memory_limit_bytes=512 * 1024**2)
    result = fq.run(program)
    print(
        dict(
            program.probabilities(result, registers=("x", "y"), max_amplitudes=1 << 24)
        )
    )
