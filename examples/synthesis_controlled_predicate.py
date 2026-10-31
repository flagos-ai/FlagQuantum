"""Synthesize a coherent control and predicate without a control-block DSL.

Run python -m examples.synthesis_controlled_predicate from the repository root.
"""

import flagquantum as fq
from flagquantum import synth


def marked(enable: int, x: int, y: int) -> int:
    return enable and (x < y)


def mask(enable: int, flag: int) -> int:
    return enable and flag


def main() -> None:
    program = synth.synthesize(
        marked,
        inputs={"enable": synth.UInt(1), "x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=16,
        memory_limit_bytes=16 * 1024**2,
    )
    result = fq.run(program)
    print(
        "control/data probabilities:",
        dict(program.probabilities(result, registers=("enable", "x", "y"))),
    )
    print("resources:", program.report["resources"])
    print("workspace:", program.report["workspace_qubits"])
    minimal = synth.synthesize(
        mask,
        inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        max_qubits=2,
        max_workspace=0,
        max_operations=3,
    )
    minimal_result = fq.run(minimal)
    print(
        "two-bit probabilities:",
        dict(minimal.probabilities(minimal_result, registers=("enable", "flag"))),
    )
    print("two-bit resources:", minimal.report["resources"])
    print("two-bit workspace:", minimal.report["workspace_qubits"])
    print(
        "Uniform probabilities do not verify controlled phase; full complex-state and interference tests do."
    )


if __name__ == "__main__":
    main()
