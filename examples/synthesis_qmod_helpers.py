"""Compose synthetic fresh-output helpers without gate or wire design.

Run python -m examples.synthesis_qmod_helpers. No Classiq SDK is needed.
"""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc prepare_grid(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
}
qfunc prepare_source(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    prepare_grid(x,y);
    energy=x*x+y;
}
qfunc main(output x:qnum, output y:qnum, output energy:qnum) {
    prepare_source(x,y,energy);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
    print(program.report["resources"])
    print("Workspace qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
