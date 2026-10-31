"""Synthesize a predicate phase without gates, physical indices or decoding.

Run python -m examples.synthesis_phase_oracle from the repository root.
Input probabilities do not reveal phase; full amplitude and interference checks
live in tests/integration/test_synthesis_public_phase.py.
"""

import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


def main() -> None:
    program = synth.synthesize(
        marked,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=22,
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    print("output mode:", program.report["output_mode"])
    print(
        "input probabilities:",
        dict(program.probabilities(result, registers=("x", "y"))),
    )
    print("resources:", program.report["resources"])
    print("workspace:", program.report["workspace_qubits"])
    print(
        "Uniform probabilities alone do not verify the phase; full amplitude and interference tests do."
    )


if __name__ == "__main__":
    main()
