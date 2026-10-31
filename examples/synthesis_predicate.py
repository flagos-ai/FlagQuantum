"""Run a coherent unsigned predicate without gates, physical indices or decoding.

Run python -m examples.synthesis_predicate from the repository root.
"""

import flagquantum as fq
from flagquantum import synth


def predicate(x: int, y: int) -> bool:
    return x * x + y < 5


def main() -> None:
    inputs = {"x": synth.UInt(2), "y": synth.UInt(2)}
    basis = synth.synthesize(
        predicate,
        inputs=inputs,
        input_state=synth.InputState.basis(x=1, y=2),
        max_qubits=22,
    )
    print(
        "basis:",
        dict(basis.probabilities(fq.run(basis), registers=("x", "y", "result"))),
    )
    coherent = synth.synthesize(
        predicate,
        inputs=inputs,
        input_state=synth.InputState.uniform(),
        max_qubits=22,
    )
    result = fq.run(coherent)
    print(
        "joint:", dict(coherent.probabilities(result, registers=("x", "y", "result")))
    )
    print("resources:", coherent.report["resources"])
    print("workspace:", coherent.report["workspace_qubits"])


if __name__ == "__main__":
    main()
