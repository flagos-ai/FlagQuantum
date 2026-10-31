"""Exact full-width probabilities for the admitted small CPU MPS domain."""

from __future__ import annotations

import torch

from .state import MPSState


def _bounded_full_width_probabilities(
    output: object, qubits: tuple[int, ...]
) -> torch.Tensor | None:
    """Return at most eight Born weights, or decline the unsupported domain.

    Each amplitude uses the existing native bond contraction. Only the requested
    real probability vector is assembled; no dense complex state is constructed.
    The caller must exclude noise and non-probability measurement requests.
    """

    if not isinstance(output, MPSState):
        return None
    n = output.n_qubits
    if (
        not 1 <= n <= 3
        or output.bsz != 1
        or output.device.type != "cpu"
        or output.dtype != torch.complex128
        or len(qubits) != n
        or set(qubits) != set(range(n))
        or output.config.cutoff != 0
        or output.config.max_bond is not None
        or any(error != 0 for error in output.truncation_errors)
    ):
        return None
    # Raw unnormalized native objects retain their existing parity semantics.
    # This is an eligibility check, never a normalization of the returned data.
    norm = output.state_norm()
    if not bool(torch.isfinite(norm).all()) or not bool(
        ((norm - 1).abs() <= 1e-12).all()
    ):
        return None

    probabilities: list[torch.Tensor] = []
    for outcome in range(1 << n):
        physical_bits = [0] * n
        for position, qubit in enumerate(qubits):
            physical_bits[qubit] = (outcome >> (n - position - 1)) & 1
        amplitude = output.amplitude(physical_bits)
        weight = amplitude.real.square() + amplitude.imag.square()
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(
                "bounded MPS probability contraction returned nonfinite data"
            )
        probabilities.append(weight)
    return torch.stack(probabilities, dim=-1)
