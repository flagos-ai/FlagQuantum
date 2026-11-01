"""Express an interval predicate without comparator gates or inverse design.

Run python -m examples.synthesis_interval from the repository root.
"""

import flagquantum as fq
from flagquantum import synth


def allowed(x: int, low: int, high: int, excluded: int) -> bool:
    return low <= x <= high and x != excluded


def main() -> None:
    for mode in ("value", "phase"):
        program = synth.synthesize(
            allowed,
            inputs={"x": synth.UInt(3)},
            constants={"low": 1, "high": 6, "excluded": 3},
            input_state=synth.InputState.uniform(),
            output_mode=mode,
            max_qubits=24,
            memory_limit_bytes=256 * 1024**2,
        )
        result = fq.run(program)
        names = ("x", "result") if mode == "value" else ("x",)
        print(
            f"{mode} probabilities:",
            dict(program.probabilities(result, registers=names)),
        )
        print(f"{mode} resources:", program.report["resources"])
        print(f"{mode} workspace:", program.report["workspace_qubits"])
    print(
        "Full complex-state tests verify the five marked signs; probabilities alone do not."
    )


if __name__ == "__main__":
    main()
