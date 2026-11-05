"""A Boolean product phase in native qfunc, Qmod and an independent Circuit."""

import torch

import flagquantum as fq
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    hadamard_transform,
    phase,
    qfunc,
    synthesize,
    within,
)


@qfunc
def main(a: Output[QBit], b: Output[QBit], c: Output[QBit], d: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    with within():
        selected: QNum[1] = a * b * c * d
    with apply():
        phase(selected, 0.37)


QMOD = """qfunc main(output a:qbit, output b:qbit, output c:qbit, output d:qbit) {
  allocate(a);allocate(b);allocate(c);allocate(d);
  H(a);H(b);H(c);H(d);
  selected:qnum<1>;
  within {selected=a*b*c*d;} apply {phase(selected,0.37);}
}"""


def circuit_reference() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    diagonal = torch.ones(16, dtype=torch.complex128)
    diagonal[-1] = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
    circuit.any(0, 1, 2, 3, matrix=torch.diag(diagonal))
    return circuit


def run_example() -> None:
    expected = fq.run(circuit_reference()).statevector().reshape(-1)
    for source in (main, QMOD):
        program = (
            synthesize(source, source_format="qmod", memory_limit_bytes=64 * 1024**2)
            if isinstance(source, str)
            else synthesize(source, memory_limit_bytes=64 * 1024**2)
        )
        actual = fq.run(program).statevector().reshape(-1)
        workspace = program.circuit.n_wires - 4
        embedded = torch.zeros_like(actual)
        embedded[:: 1 << workspace] = expected
        torch.testing.assert_close(actual, embedded, atol=1e-12, rtol=0)
        print(program.report["resources"])
    print("Native qfunc and Qmod match Circuit, including clean workspace.")


if __name__ == "__main__":
    run_example()
