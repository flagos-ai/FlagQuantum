"""Shared dense quantum probability extraction, reduction and reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

# How far a reduced total may sit from one before it is divided out. The gap is
# wide on purpose: it has to absorb the rounding a normalised state accumulates
# through many gates, which is orders of magnitude below this, while still
# catching a state that was never normalised in the first place.
_TRACE_TOLERANCE = 1e-4


def _reduced_joint_marginal(
    probabilities: torch.Tensor,
    wires: tuple[int, ...],
    *,
    n_wires: int,
) -> torch.Tensor:
    """Sum a full probability distribution down to the requested qubits.

    ``probabilities`` is one value per basis state, ``(batch, 2 ** n_wires)`` and
    big-endian per qubit, which is the convention a density matrix's diagonal and
    a statevector's squared amplitudes already share. Reducing that tensor is a
    reshape and a sum, so the cost is one pass over a distribution that is
    already in memory rather than one full-state contraction per parity term.

    The axes that survive come out in qubit order, so a request that named its
    qubits out of order is permuted back before the reshape. Getting that wrong
    would silently transpose the marginal rather than fail, which is why the
    order is restored here, in the one place every caller goes through.

    A total other than one is divided out. Every state the engine produces is
    normalised, so this leaves those runs untouched -- the division is skipped
    unless the total is off by more than ``_TRACE_TOLERANCE``, and dividing by an
    exact one is the identity anyway. It is here because a caller can hand a raw
    tensor to ``execute_measurements``, and the statevector arm used to reach
    such a state through ``expectation_ps``, which normalises as a side effect;
    without this the same call would return a scaled marginal instead of a
    distribution. Keeping the two arms on one rule is the point of the helper.
    """

    shaped = probabilities.reshape((probabilities.shape[0],) + (2,) * n_wires)
    unselected_axes = tuple(
        wire + 1 for wire in range(n_wires) if wire not in set(wires)
    )
    marginal = shaped.sum(dim=unselected_axes) if unselected_axes else shaped
    current_order = tuple(sorted(wires))
    if wires != current_order:
        permutation = (0,) + tuple(current_order.index(wire) + 1 for wire in wires)
        marginal = marginal.permute(permutation)
    marginal = marginal.reshape(probabilities.shape[0], 2 ** len(wires))

    return _normalize_joint_marginal(marginal)


def _normalize_joint_marginal(marginal: torch.Tensor) -> torch.Tensor:
    """Preserve the dense-result normalization contract for either route."""

    total = marginal.sum(dim=-1, keepdim=True)
    if marginal.is_cuda:
        normalize = ((total - 1).abs() > _TRACE_TOLERANCE).any() & (total > 0).all()
        denominator = torch.where(normalize, total, torch.ones_like(total))
        return marginal / denominator
    if bool(((total - 1).abs() > _TRACE_TOLERANCE).any()) and bool((total > 0).all()):
        marginal = marginal / total
    return marginal


def _ideal_probabilities(output: Any, n_wires: int) -> torch.Tensor | None:
    """The distribution a dense result already carries, or ``None``.

    A density matrix carries it on its diagonal and a statevector in its squared
    amplitudes. Both are read here, and neither is reconstructed: the point of
    the two tests is that the tensor in hand is already the distribution, so
    reducing it costs one sum.

    A target that is not a dense tensor -- an MPS or tensor-network result --
    returns ``None`` and keeps the parity path. That is not a preference but a
    boundary: those targets can answer ``probabilities()`` by materialising a
    dense state of ``2 ** n_wires`` amplitudes, which is the cost they exist to
    avoid, and at 24 qubits that made a 0.5 ms marginal take 330 ms.
    """

    if not isinstance(output, torch.Tensor) or not output.is_complex():
        return None
    dimension = 2**n_wires
    if output.ndim == 3 and output.shape[-2:] == (dimension, dimension):
        return torch.real(torch.diagonal(output, dim1=-2, dim2=-1))
    if output.ndim == 2 and output.shape[-1] == dimension:
        return torch.abs(output) ** 2
    return None


def _probabilities_from_parity_expectations(
    expectations: Sequence[torch.Tensor],
    *,
    n_wires: int,
) -> torch.Tensor:
    template = expectations[0]
    probabilities = []
    for outcome in range(1 << n_wires):
        value = torch.ones_like(template)
        for mask, term in enumerate(expectations, start=1):
            parity = sum(
                (outcome >> (n_wires - index - 1)) & 1
                for index in range(n_wires)
                if mask & (1 << index)
            )
            value = value + (-term if parity % 2 else term)
        probabilities.append(value / (1 << n_wires))
    return torch.stack(probabilities, dim=-1)


def _sample_pauli_distribution(
    distribution: torch.Tensor,
    *,
    shots: int,
    n_qubits: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample the Pauli readout distribution with its existing normalization."""

    indices = torch.multinomial(
        distribution,
        shots,
        replacement=True,
        generator=generator,
    )
    shifts = torch.arange(n_qubits - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.int64)


def _normalize_pauli_sampling_distribution(
    distribution: torch.Tensor,
) -> torch.Tensor:
    """Preserve the Pauli sampling clamp and row normalization order."""

    distribution = torch.clamp(distribution, min=0)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return distribution


def _statevector_readout_probabilities(state: torch.Tensor) -> torch.Tensor:
    """Read raw basis probabilities without normalization or reconstruction."""

    return torch.abs(state) ** 2


def _statevector_readout_z_expectations(
    state: torch.Tensor,
    qubits: Sequence[int],
    *,
    n_qubits: int,
    total_amplitudes: int,
) -> torch.Tensor:
    """Read the existing ordered dense-state Z moments from raw amplitudes."""

    probabilities = torch.abs(state) ** 2
    values = []
    indices = torch.arange(total_amplitudes, device=state.device)
    for target in qubits:
        mask = 1 << (n_qubits - int(target) - 1)
        signs = torch.where((indices & mask) == 0, 1.0, -1.0).to(
            device=state.device, dtype=probabilities.real.dtype
        )
        values.append((probabilities * signs.reshape(1, -1)).sum(dim=-1))
    return torch.stack(values, dim=-1)
