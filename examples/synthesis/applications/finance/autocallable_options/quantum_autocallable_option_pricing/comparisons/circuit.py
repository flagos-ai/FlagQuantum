"""Independent branch Circuits and exact rank-one amplitude-amplification reference.

The reference enumerates the original 64 classical grid branches. The canonical
state retains every source word and the exponential integration register.
Reflection is evaluated as a rank-one action, without a dense 2048-square
reference matrix. No native construction or lowering is used.
"""

import math
from itertools import product

import torch

import flagquantum as fq

from ..physics import DISTRIBUTION


def branch_circuit(
    steps: tuple[int, ...], *, shift: float, marker: int = 0
) -> fq.Circuit:
    """Evaluate signed, coefficient-truncated source payoffs independently."""
    offset = math.trunc((shift - 3 * 0.238268134860707) * 4)
    slope = math.trunc(0.238268134860707 * 1.5 * 4)
    increments = tuple(offset + slope * word for word in steps)
    first = increments[0]
    second = sum(increments[:2])
    final = sum(increments)
    circuit = fq.Circuit(5, dtype=torch.complex128)
    if marker == 1:
        circuit.x(4)
    elif marker == 2:
        circuit.ry(4, 0.73).rz(4, 0.41)
    elif marker != 0:
        raise ValueError("Marker preparation must be 0, 1 or 2")
    if first > 0:
        return circuit.ry(4, 0.875426634895785)
    if second > 0:
        return circuit.ry(4, 0.89966916778117)
    if min(first, second, final) >= -1 or final >= 0:
        return circuit.ry(4, 0.818547457426368)
    for bit in range(4):
        circuit.ry(3 - bit, 2 * math.atan(math.exp(0.125 * (1 << bit))))
    permutation = torch.eye(32, dtype=torch.complex128)
    for word in range(final + 8 + 1):
        low = 2 * word
        permutation[[low, low + 1]] = permutation[[low + 1, low]]
    return circuit.any(*range(5), matrix=permutation)


def prepared_state(*, shift: float = 0.1274075, marker: int = 0) -> torch.Tensor:
    """Return canonical (indicator, x0, x1, x2, reference) amplitudes."""
    branches = []
    for steps in product(range(4), repeat=3):
        amplitude = math.sqrt(math.prod(DISTRIBUTION[word] for word in steps))
        state = (
            fq.run(branch_circuit(steps, shift=shift, marker=marker))
            .statevector()
            .flatten()
        )
        branches.append(amplitude * state)
    logical = torch.stack(branches).reshape(4, 4, 4, 16, 2)
    return logical.permute(4, 0, 1, 2, 3).flatten()


def reference_state(k: int = 1, *, shift: float = 0.1274075) -> torch.Tensor:
    """Apply the exact oracle and source-state reflection to every amplitude.

    Independent five-bit branch Circuits supply the prepared purification. The
    Grover reflection 2|prepared><prepared| - I preserves its absolute sign.
    This reference action needs O(2**11) memory instead of a dense unitary.
    """
    if type(k) is not int or not 0 <= k <= 4:
        raise ValueError("Iteration count must be an integer in 0..4")
    prepared = prepared_state(shift=shift)
    state = prepared.clone()
    for _ in range(k):
        state[1024:] *= -1
        state = 2 * torch.vdot(prepared, state) * prepared - state
    return state


def run_example() -> None:
    from ..main import build_program

    for k in (0, 1, 2):
        program = build_program(k)
        actual = fq.run(program).statevector().reshape(2048, -1)
        expected = reference_state(k)
        torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
        if float(actual[:, 1:].abs().square().sum()) >= 1e-20:
            raise AssertionError("Synthesis workspace did not return to zero")
        print(k, program.report["resources"])
    print("Complete purification, absolute phase and clean workspace agree.")


if __name__ == "__main__":
    run_example()
