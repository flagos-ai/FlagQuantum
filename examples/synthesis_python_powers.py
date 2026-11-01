"""Use a positive literal power in ordinary high-level Python logic."""

import flagquantum as fq
from flagquantum import synth


def cube(x: int) -> int:
    return x**3


def main() -> None:
    program = synth.synthesize(
        cube,
        inputs={"x": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "result"))))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
