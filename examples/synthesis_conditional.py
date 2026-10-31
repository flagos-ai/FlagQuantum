"""Select unsigned values and phase predicates without designing quantum gates.

Run python -m examples.synthesis_conditional from the repository root.
"""

import flagquantum as fq
from flagquantum import synth


def smaller(x: int, y: int) -> int:
    return x if x < y else y


def marked(enable: int, x: int, y: int) -> bool:
    return (x < y) if enable else (x == y)


def main() -> None:
    program = synth.synthesize(
        smaller,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        max_qubits=20,
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    print(
        "minimum probabilities:",
        dict(program.probabilities(result, registers=("x", "y", "result"))),
    )
    print("minimum resources:", program.report["resources"])
    print("minimum workspace:", program.report["workspace_qubits"])
    phase = synth.synthesize(
        marked,
        inputs={"enable": synth.UInt(1), "x": synth.UInt(1), "y": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=20,
        memory_limit_bytes=64 * 1024**2,
    )
    print(
        "phase probabilities:",
        dict(phase.probabilities(fq.run(phase), registers=("enable", "x", "y"))),
    )
    print("phase resources:", phase.report["resources"])
    print("phase workspace:", phase.report["workspace_qubits"])
    print("Phase acceptance compares complex amplitudes, not just probabilities.")


if __name__ == "__main__":
    main()
