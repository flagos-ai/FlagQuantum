"""Independent matrix construction of the complete molecular walk sequence."""

import math

import torch

import flagquantum as fq

from ..physics import PAULI_WORDS, PROBABILITIES, SELECTOR_PHASES


def walk_matrix() -> torch.Tensor:
    """Build the signed selector and its preparation conjugation independently."""
    dtype = torch.complex128
    paulis = {
        "I": torch.eye(2, dtype=dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=dtype),
    }
    selected = []
    for word, angle in zip(PAULI_WORDS, SELECTOR_PHASES, strict=True):
        matrix = torch.ones((1, 1), dtype=dtype)
        for axis in reversed(word):
            matrix = torch.kron(matrix, paulis[axis])
        selected.append(complex(math.cos(angle), math.sin(angle)) * matrix)

    # Reconstruct conditional rotations from probabilities, rather than native
    # instructions or loading angles. The preparation's extension off |0> is
    # part of the walk, so an arbitrary state-loading matrix would be incorrect.
    preparation = torch.eye(16, dtype=dtype)
    for level in range(4):
        length = 16 >> level
        half = length // 2
        for start in range(0, 16, length):
            left = math.fsum(PROBABILITIES[start : start + half])
            right = math.fsum(PROBABILITIES[start + half : start + length])
            if left + right == 0:
                continue
            cosine = math.sqrt(left / (left + right))
            sine = math.sqrt(right / (left + right))
            a = preparation[start : start + half].clone()
            b = preparation[start + half : start + length].clone()
            preparation[start : start + half] = cosine * a - sine * b
            preparation[start + half : start + length] = sine * a + cosine * b
    distribution = torch.kron(preparation, torch.eye(16, dtype=dtype))
    reflection = torch.diag(torch.tensor([1] * 16 + [-1] * 240, dtype=dtype))
    selector = torch.zeros((256, 256), dtype=dtype)
    for index, matrix in enumerate(selected):
        selector[16 * index : 16 * (index + 1), 16 * index : 16 * (index + 1)] = matrix
    return reflection @ distribution.mH @ selector @ distribution


def reference_circuit() -> fq.Circuit:
    """Keep the source's conjugated powers, private data and absolute phase."""
    dtype = torch.complex128
    block = tuple(range(4))
    phase = tuple(range(4, 9))
    data = tuple(range(9, 13))
    circuit = fq.Circuit(13, dtype=dtype).x(data[3]).x(data[1])
    for wire in phase:
        circuit.h(wire)
    walk = walk_matrix()
    controlled_walk = torch.zeros((512, 512), dtype=dtype)
    controlled_walk[:256, :256] = torch.eye(256, dtype=dtype)
    controlled_walk[256:, 256:] = walk
    circuit.any(
        phase[-1],
        *block,
        *data,
        matrix=controlled_walk,
    )
    signs = torch.ones(32, dtype=dtype)
    signs[0] = -1
    controlled_reflection = torch.diag(signs)
    for index in range(4):
        wire = phase[-2 - index]
        circuit.any(wire, *block, matrix=controlled_reflection)
        circuit.any(*block, *data, matrix=torch.linalg.matrix_power(walk, 2**index))
        circuit.any(wire, *block, matrix=controlled_reflection)
    words = torch.arange(32, dtype=torch.float64)
    inverse_qft = torch.exp(-2j * math.pi / 32 * torch.outer(words, words)) / math.sqrt(
        32
    )
    return circuit.any(*phase, matrix=inverse_qft)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    state = fq.run(program).statevector().reshape(8192, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=1e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-20
    print(program.report["resources"])
    print("Complete molecular walk amplitudes and clean workspace agree.")


if __name__ == "__main__":
    run_example()
