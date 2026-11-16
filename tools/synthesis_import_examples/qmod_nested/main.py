"""Compute a sum and phase a temporary condition with lexical cleanup."""

import flagquantum as fq
from flagquantum import synth

SOURCE = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    energy:qnum; flag:qbit;
    within {energy=x+y;} apply {
        within {flag=energy<4;} apply {phase(flag,pi);}
    }
}"""


def main() -> None:
    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=24)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])


if __name__ == "__main__":
    main()
