"""A synthetic native sign phase, followed by interference and named reading.

Run python -m examples.synthesis_qmod_phase. No Classiq SDK is needed.
"""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<1>, output y:qnum<1>) {
    allocate(x);
    allocate(y);
    hadamard_transform(x);
    hadamard_transform(y);
    phase(x*y, pi);
    hadamard_transform(y);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])
    print("Workspace qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
