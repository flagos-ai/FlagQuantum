"""Compose a coherent program while the compiler shares repeated arithmetic."""

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= (x + y) * (y + x)


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(1, x)
    allocate(1, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


QMOD = """qfunc energy(const x:qnum<1>,const y:qnum<1>,output result:qnum) {
    result=(x+y)*(y+x);
}
qfunc main(output x:qnum<1>,output y:qnum<1>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}"""


def circuit_reference() -> fq.Circuit:
    """Independent one-bit reference; output bits are xy, zero, x XOR y."""
    return (
        fq.Circuit(5, dtype=torch.complex128).h(0).h(1).cx(0, 4).cx(1, 4).ccx(0, 1, 2)
    )


def run() -> None:
    program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
    result = fq.run(program)
    outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, (x + y) ** 2): 1 / 4 for x in range(2) for y in range(2)}
    assert outcomes.keys() == expected.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for key, value in expected.items())
    other = synth.synthesize(
        QMOD, source_format="qmod", memory_limit_bytes=64 * 1024**2
    )
    assert (
        dict(other.probabilities(fq.run(other), registers=("x", "y", "result")))
        == outcomes
    )
    print(outcomes)
    print(program.report["resources"])
    print("Workspace:", program.report["workspace_qubits"])
    print("Repeated expressions share only within a verified pure arithmetic segment.")


if __name__ == "__main__":
    run()
