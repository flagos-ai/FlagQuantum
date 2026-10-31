"""Build static sums and predicates without manual unrolling, gates or decoding.

Run python -m examples.synthesis_static_loop from the repository root.
"""

import flagquantum as fq
from flagquantum import synth


def weighted(x: int, y: int, steps: int) -> int:
    total = x
    for i in range(steps):
        term = y * i
        total = total + term
    return total


def matched(x: int, y: int, steps: int) -> bool:
    flag = x == y
    for value in range(1, steps):
        flag = flag or (x == value)
    return flag


def main() -> None:
    weighted_program = synth.synthesize(
        weighted,
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        constants={"steps": 3},
        input_state=synth.InputState.uniform(),
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    weighted_result = fq.run(weighted_program)
    print(
        "weighted probabilities:",
        dict(
            weighted_program.probabilities(
                weighted_result, registers=("x", "y", "result")
            )
        ),
    )
    print("weighted resources:", weighted_program.report["resources"])
    print("weighted workspace:", weighted_program.report["workspace_qubits"])
    phase_program = synth.synthesize(
        matched,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        constants={"steps": 3},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=24,
        memory_limit_bytes=256 * 1024**2,
    )
    phase_result = fq.run(phase_program)
    print(
        "phase input probabilities:",
        dict(phase_program.probabilities(phase_result, registers=("x", "y"))),
    )
    print("phase resources:", phase_program.report["resources"])
    print("phase workspace:", phase_program.report["workspace_qubits"])
    print(
        "Uniform probabilities alone do not verify phase; full amplitude acceptance does."
    )


if __name__ == "__main__":
    main()
