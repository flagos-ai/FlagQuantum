"""Fresh periodic finite-difference matrices in the original axis order."""

import numpy as np
import numpy.typing as npt


def laplacian_1d(size: int) -> npt.NDArray[np.complex128]:
    """Return (S-minus - 2I + S-plus)/4 at the declared periodic size."""
    if type(size) is not int or not 2 <= size <= 32 or size & (size - 1):
        raise ValueError("grid size must be a power of two in 2..32")
    shift = np.zeros((size, size), dtype=np.complex128)
    for column in range(size):
        shift[(column + 1) % size, column] = 1
    return np.asarray(
        (shift.conj().T - 2 * np.eye(size) + shift) / 4, dtype=np.complex128
    )


def laplacian_nd(sizes: tuple[int, ...]) -> npt.NDArray[np.complex128]:
    """Retain the source reversed Kronecker order and power-of-two selector scale."""
    if not sizes or len(sizes) > 4:
        raise ValueError("one to four periodic dimensions are supported")
    matrices = tuple(laplacian_1d(size) for size in sizes)
    total = int(np.prod(sizes))
    if total > 256:
        raise ValueError("bounded matrix reference supports at most 256 grid points")
    result = np.zeros((total, total), dtype=np.complex128)
    for axis in range(len(sizes)):
        factors = tuple(
            matrices[index] if index == axis else np.eye(size, dtype=np.complex128)
            for index, size in enumerate(sizes)
        )
        term = factors[-1]
        for factor in reversed(factors[:-1]):
            term = np.asarray(np.kron(term, factor), dtype=np.complex128)
        result += term
    return result / (1 << (len(sizes) - 1).bit_length())
