"""Mark an unsigned predicate without declaring a temporary flag."""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    phase(x<y,pi);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=9)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
