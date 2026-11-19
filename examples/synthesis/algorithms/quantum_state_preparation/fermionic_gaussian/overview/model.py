"""Fresh orbital models and the bounded Bogoliubov vacuum calculation."""

import math
from dataclasses import dataclass

import torch

REAL = torch.float64
COMPLEX = torch.complex128


def periodic_hamiltonian() -> torch.Tensor:
    """Source four-site ring; spin sectors are interleaved."""
    sites = 4
    result = torch.zeros((8, 8), dtype=REAL)
    for site in range(sites):
        for spin in range(2):
            first = 2 * site + spin
            second = 2 * ((site + 1) % sites) + spin
            result[first, second] = result[second, first] = -1
        result[2 * site, 2 * site] = -10 * math.exp(
            -((site - 2) ** 2) / (2 * (4 / 6) ** 2)
        )
    return result


def pairing_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[1, 0.5], [0.5, -0.5]], dtype=COMPLEX), torch.tensor(
        [[0, 0.3], [-0.3, 0]], dtype=COMPLEX
    )


@dataclass(frozen=True)
class Canonical:
    generator: torch.Tensor
    transform: torch.Tensor
    blocks: torch.Tensor
    bogoliubov: torch.Tensor
    energies: torch.Tensor


def canonicalize(h: torch.Tensor, delta: torch.Tensor) -> Canonical:
    """Build the real skew-Schur planes from Hermitian iA eigenvectors."""
    if h.shape != (2, 2) or delta.shape != (2, 2):
        raise ValueError("this source pairing example requires exactly two modes")
    h, delta = h.to(COMPLEX), delta.to(COMPLEX)
    if not (
        torch.isfinite(h).all()
        and torch.isfinite(delta).all()
        and torch.allclose(h, h.mH, atol=1e-13, rtol=0)
        and torch.allclose(delta, -delta.T, atol=1e-13, rtol=0)
    ):
        raise ValueError("expect finite Hermitian hopping and antisymmetric pairing")
    eye = torch.eye(2, dtype=COMPLEX)
    omega = torch.cat(
        (torch.cat((eye, eye), dim=1), torch.cat((1j * eye, -1j * eye), dim=1)),
        dim=0,
    ) / math.sqrt(2)
    matrix = torch.cat(
        (torch.cat((delta, h), dim=1), torch.cat((-h.conj(), -delta.conj()), dim=1)),
        dim=0,
    )
    generator = (-1j * omega.conj() @ matrix @ omega.mH).real
    torch.testing.assert_close(generator, -generator.T, atol=1e-13, rtol=0)
    spectrum, vectors = torch.linalg.eigh(1j * generator)
    energies = spectrum[2:]
    if float(energies.min()) <= 1e-12:
        raise ValueError("the paired vacuum requires a nonzero canonical gap")
    positive = vectors[:, 2:]
    transform = torch.cat((positive.real.T, -positive.imag.T), dim=0) * math.sqrt(2)
    blocks = torch.zeros_like(generator)
    blocks[:2, 2:] = torch.diag(energies)
    blocks[2:, :2] = -torch.diag(energies)
    torch.testing.assert_close(
        transform @ transform.T, torch.eye(4, dtype=REAL), atol=1e-13, rtol=0
    )
    torch.testing.assert_close(
        transform @ generator @ transform.T, blocks, atol=1e-13, rtol=0
    )
    bogoliubov = omega.mH @ transform.to(COMPLEX) @ omega
    torch.testing.assert_close(
        bogoliubov @ bogoliubov.mH, torch.eye(4, dtype=COMPLEX), atol=1e-13, rtol=0
    )
    return Canonical(generator, transform, blocks, bogoliubov, energies)


def vacuum(state: Canonical) -> tuple[torch.Tensor, float]:
    """Find the common kernel of the two quasiparticle annihilators."""
    lower = torch.zeros((2, 4, 4), dtype=COMPLEX)
    lower[0, 0, 1] = lower[0, 2, 3] = lower[1, 0, 2] = 1
    lower[1, 1, 3] = -1
    operators = torch.cat((lower.mH, lower), dim=0)
    annihilators = state.bogoliubov[2:] @ operators.reshape(4, 16)
    matrix = annihilators.reshape(8, 4)
    _, values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if float(values[-2]) <= 1e-12 or float(values[-1]) > 1e-12:
        raise ValueError("the two quasiparticle annihilators must have a unique vacuum")
    vector = vh[-1].conj()
    return vector, float((matrix @ vector).abs().max())
