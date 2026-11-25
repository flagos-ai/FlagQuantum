"""Run a range-proven constant addition without allocating carry wires."""

import flagquantum as fq
from flagquantum import synth


def offset(x: int) -> int:
    return x + 256


def main() -> None:
    program = synth.synthesize(
        offset,
        inputs={"x": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "result"))))
    print(program.report["resources"])
    print(f"Planned state buffer: {program.execution.state_bytes} bytes")


if __name__ == "__main__":
    main()
