"""Prepare and compute in one Python qfunc, with Qmod and Circuit references."""

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def prepare(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x * x + y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    prepare(x, y)
    energy(x, y, result)


QMOD = """qfunc energy(const x:qnum<2>,const y:qnum<2>,output result:qnum) {
    result=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}"""


def circuit_reference(*, uniform: bool = True) -> fq.Circuit:
    """Independent two-bit Boolean derivation; one clean scratch bit.

    Wires 0/1 encode x, 2/3 encode y, 4..7 encode the result (MSB first).
    The final scratch wire stores b*c transiently for the carry into bit two.
    This domain-specific reference is not the general synthesis implementation.
    """
    circuit = fq.Circuit(9, dtype=torch.complex128)
    if uniform:
        for wire in range(4):
            circuit.h(wire)
    circuit.cx(1, 7).cx(3, 7)
    circuit.cx(2, 6).ccx(1, 3, 6)
    circuit.x(1).ccx(0, 1, 5).x(1)
    circuit.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8)
    circuit.ccx(0, 1, 4)
    return circuit


def run() -> None:
    program = synth.synthesize(main, max_qubits=24, memory_limit_bytes=2**30)
    result = fq.run(program)
    outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert outcomes.keys() == expected.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for key, value in expected.items())
    migrated = synth.synthesize(QMOD, source_format="qmod", memory_limit_bytes=2**30)
    other = dict(
        migrated.probabilities(fq.run(migrated), registers=("x", "y", "result"))
    )
    assert outcomes.keys() == other.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for key, value in other.items())
    reference = fq.run(circuit_reference()).statevector().reshape(4, 4, 16, 2)
    oracle = torch.zeros_like(reference)
    for x in range(4):
        for y in range(4):
            oracle[x, y, x * x + y, 0] = 0.25
    torch.testing.assert_close(reference, oracle, atol=1e-12, rtol=0)
    print(outcomes)
    print(program.report["resources"])
    print("Workspace:", program.report["workspace_qubits"])
    print("Python in-program preparation and Qmod agree on all 16 outcomes.")


if __name__ == "__main__":
    run()
