"""Cube a coherent unsigned input through existing multiplication synthesis."""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<2>,output result:qnum) {
    allocate(x); hadamard_transform(x); result=x**3;
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "result"))))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
