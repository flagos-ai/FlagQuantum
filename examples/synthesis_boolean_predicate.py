"""Compose coherent conditions without gates, wire indices or cleanup code.

Run python -m examples.synthesis_boolean_predicate from the repository root.
"""

from typing import Literal

import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return (x < 2 and y == 1) or not (x == y)


def main() -> None:
    modes: tuple[Literal["value", "phase"], ...] = ("value", "phase")
    for mode in modes:
        program = synth.synthesize(
            marked,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            input_state=synth.InputState.uniform(),
            output_mode=mode,
            max_qubits=24,
            memory_limit_bytes=64 * 1024**2,
        )
        result = fq.run(program)
        registers = ("x", "y", "result") if mode == "value" else ("x", "y")
        print(
            f"{mode} probabilities:",
            dict(program.probabilities(result, registers=registers)),
        )
        print(f"{mode} resources:", program.report["resources"])
        print(f"{mode} workspace:", program.report["workspace_qubits"])
    print(
        "Uniform probabilities alone do not verify phase; complete amplitude and interference tests do."
    )


if __name__ == "__main__":
    main()
