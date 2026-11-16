"""Coherent unsigned comparison from native source, with named reading.

Run python -m tools.synthesis_import_examples.qmod_comparisons.main. No Classiq SDK is needed.
"""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc compare(const x:qnum<2>, const y:qnum<2>, output flag:qbit) {
    flag=x<y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output flag:qbit) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    compare(x,y,flag);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y", "flag"))))
    print(program.report["resources"])
    print("Workspace qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
