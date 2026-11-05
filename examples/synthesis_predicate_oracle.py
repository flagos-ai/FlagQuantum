"""One predicate-driven Grover round in Qmod, native qfunc and public Circuit."""

import math

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
def mark(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, math.pi)


@qfunc
def reflect_nonzero(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, math.pi)
    phase(a, -math.pi)
    phase(b, -math.pi)


@qfunc
def main(a: Output[QBit], b: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    mark(a, b)
    hadamard_transform(a)
    hadamard_transform(b)
    reflect_nonzero(a, b)
    hadamard_transform(a)
    hadamard_transform(b)


QMOD = """qperm pred(const data:qbit[2],marker:qbit) {
  marker ^= data[0] & data[1];
}
qfunc main(output q:qbit[2]) {
  allocate(q);
  grover_search(1,lambda(target) {phase_oracle(pred,target);},q);
}"""


def run_example() -> None:
    diffuser = torch.full((4, 4), 0.5, dtype=torch.complex128) - torch.eye(
        4, dtype=torch.complex128
    )
    circuit = (
        fq.Circuit(2, dtype=torch.complex128).h(0).h(1).cphase(0, 1, theta=math.pi)
    )
    circuit.any(0, 1, matrix=diffuser)
    reference = fq.run(circuit).statevector().reshape(-1)
    for program in (
        synth.synthesize(QMOD, source_format="qmod"),
        synth.synthesize(main),
    ):
        actual = fq.run(program).statevector().reshape(-1)
        expected = torch.zeros_like(actual)
        expected[:: 1 << (program.circuit.n_wires - 2)] = reference
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print(
        "Predicate Qmod, native qfunc intent and Circuit agree, including global phase and clean scratch."
    )


if __name__ == "__main__":
    run_example()
