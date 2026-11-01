"""Native Const arithmetic through the existing checked Program and named reader.

Run python -m examples.synthesis_qmod_const. No Classiq SDK is needed.
"""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc compute_energy(const x:qnum<2>, const y:qnum<2>, output energy:qnum) {
    energy=x*x+y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    compute_energy(x,y,energy);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
    print(program.report["resources"])
    print("Workspace qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
