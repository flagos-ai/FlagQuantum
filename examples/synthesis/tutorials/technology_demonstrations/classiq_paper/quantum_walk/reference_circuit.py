"""Independent dense permutation oracle for seven logical reference-walk bits."""

import torch

import flagquantum as fq


def permutation(*, catalyst: bool) -> tuple[int, ...]:
    """Derive the source Boolean cascade without quantum synthesis or QFT."""
    if not catalyst:
        return tuple((word + 1) % 64 for word in range(64))
    image: list[int] = []
    for word in range(64):
        bits = [(word >> index) & 1 for index in range(6)]
        for index in range(5, 0, -1):
            bits[index] ^= int(all(bits[:index]))
            bits[0] ^= 1
        image.append(sum(bit << index for index, bit in enumerate(bits)))
    if sorted(image) != list(range(64)):
        raise ValueError("Source shift is not reversible.")
    return tuple(image)


def reference_circuit(*, catalyst: bool = False) -> fq.Circuit:
    """Build H then an independent 128-column controlled permutation.

    The 128x128 complex128 matrix occupies 256 KiB and is used for correctness
    only. It is not the native lowering or a resource/performance baseline.
    """
    image = permutation(catalyst=catalyst)
    inverse = tuple(image.index(word) for word in range(64))
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for word in range(64):
        matrix[2 * inverse[word], 2 * word] = 1
        matrix[2 * image[word] + 1, 2 * word + 1] = 1
    return fq.Circuit(7, dtype=torch.complex128).h(6).any(*range(7), matrix=matrix)
