"""Independent transform definitions derived from complex Fourier coefficients."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq


def transform_matrix(width: int, name: str) -> torch.Tensor:
    """Build the reference matrix without calling synthesis or its lowering."""
    size = 1 << width
    samples = torch.arange(size, dtype=torch.float64)
    if name == "qct_qst_type1":
        fourier = torch.exp(2j * math.pi / size * torch.outer(samples, samples))
        fourier /= math.sqrt(size)
        basis = torch.eye(size, dtype=torch.complex128)
        half = size // 2
        even = [(basis[j] + basis[-j]) / math.sqrt(2) for j in range(1, half)]
        odd = [(basis[j] - basis[-j]) / math.sqrt(2) for j in range(1, half)]
        parity = torch.stack([basis[0], *even, basis[half], *odd], dim=1)
        return parity.mH @ fourier @ parity
    modes = torch.arange(size + 1, dtype=torch.float64)
    coefficients = torch.exp(1j * math.pi * torch.outer(samples + 0.5, modes) / size).T
    cosine = coefficients[:-1].real.clone()
    sine = coefficients[1:].imag.clone()
    cosine[0] *= 2**-0.5
    sine[-1] *= 2**-0.5
    scale = math.sqrt(2 / size)
    if name == "qct_qst_type2":
        zeros = torch.zeros_like(cosine)
        return scale * torch.cat(
            (torch.cat((cosine, zeros), dim=1), torch.cat((zeros, -sine), dim=1))
        ).to(torch.complex128)
    if name not in {"qct_type2", "qst_type2"}:
        raise ValueError("unknown reference transform")
    return scale * (sine if name == "qst_type2" else cosine).to(torch.complex128)


def transform_circuit(amplitudes: Sequence[float], name: str) -> fq.Circuit:
    """Prepare the retained vector with a Householder reflection, then transform."""
    width = len(amplitudes).bit_length() - 1
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    vector /= vector.norm()
    basis = torch.eye(len(vector), dtype=torch.complex128)
    delta = basis[:, 0] - vector
    prepare = basis
    if delta.norm() > 1e-14:
        prepare = basis - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    combined = name == "qct_qst_type2"
    circuit = fq.Circuit(width + int(combined), dtype=torch.complex128)
    circuit.any(*range(width), matrix=prepare)
    if combined:
        circuit.h(width)
        circuit.any(width, *range(width), matrix=transform_matrix(width, name))
    else:
        circuit.any(*range(width), matrix=transform_matrix(width, name))
    return circuit
