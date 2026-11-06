"""A repeated phase cost in Qmod, native qfunc and an independent Circuit.

The Qmod entry binds an exact integer. The equivalent native qfunc uses two
static helper calls; its existing Python parameter contract is unchanged.
"""

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def twist(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, 0.6)


@qfunc
def main(a: Output[QBit], b: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    twist(a, b)
    twist(a, b)


QMOD = """qfunc twist(steps:int, q:qbit[2]) {
  power(steps) {phase(0.6*q[0]*q[1]);}
}
qfunc main(output q:qbit[2]) {
  allocate(q);hadamard_transform(q);twist(2,q);
}"""


def run_example() -> None:
    reference = (
        fq.run(fq.Circuit(2, dtype=torch.complex128).h(0).h(1).cphase(0, 1, theta=1.2))
        .statevector()
        .reshape(-1)
    )
    programs = (
        synth.synthesize(QMOD, source_format="qmod"),
        synth.synthesize(main),
    )
    for program in programs:
        state = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 2
        expected = torch.zeros_like(state)
        expected[:: 1 << workspace] = reference
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Qmod integer composition, static native qfunc and Circuit agree, including global phase."
    )


if __name__ == "__main__":
    run_example()
