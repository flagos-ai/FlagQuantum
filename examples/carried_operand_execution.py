"""Run a gate that carries its own matrix instead of the operator its name means.

`Circuit.gate(name, qubits, matrix=...)` attaches an operand: from that point on
the instruction is the matrix the caller wrote, and `name` is only the spelling it
was registered under. Three executors used to choose a kernel from the name before
they read the operand, so `x` applied `X`, `y` applied `Y`, and `z` applied `Z`
whatever the caller had attached -- and the identity pass deleted a
`rz(theta=0.0)` that carried one, because a zero angle is what it reads.

This example makes the difference visible without measuring anything subtle. The
operand is a real reflection whose image of `|0>` is far from every image the
named operators have, so a route that substitutes the named operator is wrong by
about a whole amplitude rather than by a rounding difference.

Runtime mode: local, single process, CPU statevector. Nothing here is distributed,
replicated, or sharded, and no device is contacted.
"""

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.experimental.dynamic import run_dynamic

#: A real symmetric unitary. It is not `X`, not `Z`, and not a phase, so no named
#: operator can stand in for it.
_OPERAND = torch.tensor(
    [[0.28, 0.96], [0.96, -0.28]],
    dtype=torch.complex128,
)

#: The two named operators the operand is confused with, written out so the three
#: images of `|0>` can be read side by side.
_PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

#: `theta` is a declared parameter of `rz` and the point the identity rule reads.
_ZERO_ANGLE = {"theta": 0.0}


def _amplitudes(state: torch.Tensor) -> str:
    """The first two amplitudes of a one-qubit state, as text."""

    values = torch.as_tensor(state).to(torch.complex128).reshape(-1)[:2]
    return "  ".join(f"{complex(value):+.4f}" for value in values)


def _register(state: torch.Tensor) -> str:
    """The eight amplitudes of a three-qubit state, as text."""

    values = torch.as_tensor(state).to(torch.complex128).reshape(-1)[:8]
    return "  ".join(f"{complex(value):+.3f}" for value in values)


def main() -> None:
    start = torch.zeros(2, dtype=torch.complex128)
    start[0] = 1.0
    print("the operand:")
    for row, entries in enumerate(_OPERAND.tolist()):
        print(f"  row {row}: {entries}")
    print("  operand |0>:", _amplitudes(_OPERAND @ start))
    print(
        "  x |0>      :",
        _amplitudes(_PAULI_X @ start),
    )
    print(
        "  z |0>      :",
        _amplitudes(_PAULI_Z @ start),
    )

    for name, params in (("x", {}), ("y", {}), ("z", {}), ("rz", _ZERO_ANGLE)):
        circuit = fq.Circuit(1)
        circuit.gate(name, [0], params=params, matrix=_OPERAND)
        print(f"{name} carrying the operand:")
        print("  circuit.state():", _amplitudes(circuit.state()))
        print(
            "  planned instructions:",
            sum(len(layer.instructions) for layer in fq.plan(circuit).layers),
        )
        dynamic = DynamicCircuit(1)
        dynamic.gate(name, [0], params=params, matrix=_OPERAND)
        final = run_dynamic(dynamic, shots=1, seed=11).final_states[0]
        print("  run_dynamic():", _amplitudes(final))

    # The controls. Without an operand the same opcode is the operator its name
    # denotes, and a zero-angle rotation is a rotation of nothing.
    plain = fq.Circuit(1)
    plain.x(0)
    print("x with no operand:")
    print("  circuit.state():", _amplitudes(plain.state()))

    zero = fq.Circuit(1)
    zero.rz(0, theta=0.0)
    print("rz(0.0) with no operand:")
    print(
        "  planned instructions:",
        sum(len(layer.instructions) for layer in fq.plan(zero).layers),
    )

    # The operand is not confined to one qubit. `any` is how a caller attaches a
    # matrix to something the schema does not declare, and it has always executed
    # as written, which is the route the four above now agree with.
    two = fq.Circuit(2)
    two.any(0, 1, matrix=torch.kron(_OPERAND, _OPERAND))
    print("the same operand on two qubits through any():")
    print(
        "  planned instructions:",
        sum(len(layer.instructions) for layer in fq.plan(two).layers),
    )
    print("  circuit.state():", _register(two.state()))


if __name__ == "__main__":
    main()
