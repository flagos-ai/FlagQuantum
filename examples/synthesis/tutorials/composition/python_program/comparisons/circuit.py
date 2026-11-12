"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.composition.python_program.main import main
from flagquantum import synth


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
    assert all(abs(outcomes[key] - value) < 1e-12 for (key, value) in expected.items())
    reference = fq.run(circuit_reference()).statevector().reshape(4, 4, 16, 2)
    state = result.statevector().reshape(4, 4, 16, -1)
    torch.testing.assert_close(state[..., 0], reference[..., 0], atol=1e-12, rtol=0)
    assert float(state[..., 1:].abs().square().sum()) < 1e-24
    oracle = torch.zeros_like(reference)
    for x in range(4):
        for y in range(4):
            oracle[x, y, x * x + y, 0] = 0.25
    torch.testing.assert_close(reference, oracle, atol=1e-12, rtol=0)
    print(outcomes)
    print(program.report["resources"])
    print("Workspace:", program.report["workspace_qubits"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run()
