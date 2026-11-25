"""Combine unsigned conditions without explicit temporary flags."""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    phase((x<2)|(y==3),pi);
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
