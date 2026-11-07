"""Native compute/apply/cleanup with hidden temporary flags and named results.

Run python -m examples.synthesis.tutorials.qmod.qmod_within.main. No Classiq SDK is needed.
"""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc mark(const x:qnum<2>, const y:qnum<2>) {
    flag:qbit;
    within { flag=x<y; } apply { phase(flag,pi); }
    within { flag=x==y; } apply { phase(flag,pi); }
}
qfunc main(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    mark(x,y);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])
    print("All internal work qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()
