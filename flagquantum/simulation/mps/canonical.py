"""Local mixed-canonical center movement for matrix-product states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .canonical_transfer_dispatch import (
    absorb_left_canonical_transfer,
    absorb_right_canonical_transfer,
)

if TYPE_CHECKING:
    from .state import MPSState


def move_orthogonality_center(state: MPSState, site: int) -> None:
    """Move ``state``'s mixed-canonical center to ``site`` in place."""

    site = int(site)
    if site < 0 or site >= state.n_qubits:
        raise ValueError(
            f"Orthogonality center must be in [0, {state.n_qubits - 1}], got {site}."
        )
    if state.orthogonality_center is None or not state._canonical_center_valid:
        state.orthogonalize_left()
    center = state.orthogonality_center
    if center is None:
        raise RuntimeError("MPS orthogonalization did not establish a center")
    if center < site:
        sweep_center_right(state, site)
    elif center > site:
        sweep_center_left(state, site)


def sweep_center_right(state: MPSState, target_site: int) -> None:
    """Apply batched QR steps from the current center toward the right."""

    center = state.orthogonality_center
    if center is None:
        center = 0
    for qubit in range(int(center), int(target_site)):
        tensor = state.tensors[qubit]
        batch, left_dim, physical_dim, right_dim = tensor.shape
        matrix = tensor.reshape(batch, left_dim * physical_dim, right_dim)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        new_right_dim = q.shape[-1]
        state.tensors[qubit] = q.reshape(
            batch,
            left_dim,
            physical_dim,
            new_right_dim,
        )
        state.tensors[qubit + 1] = absorb_left_canonical_transfer(
            r, state.tensors[qubit + 1]
        )
    state.orthogonality_center = int(target_site)
    state._canonical_center_valid = True


def sweep_center_left(state: MPSState, target_site: int) -> None:
    """Apply batched QR steps from the current center toward the left."""

    center = state.orthogonality_center
    if center is None:
        center = state.n_qubits - 1
    for qubit in range(int(center), int(target_site), -1):
        tensor = state.tensors[qubit]
        batch, left_dim, physical_dim, right_dim = tensor.shape
        matrix = tensor.reshape(batch, left_dim, physical_dim * right_dim)
        q, r = torch.linalg.qr(matrix.transpose(-1, -2), mode="reduced")
        right_orthogonal = q.transpose(-1, -2)
        transfer = r.transpose(-1, -2)
        new_left_dim = right_orthogonal.shape[1]
        state.tensors[qubit] = right_orthogonal.reshape(
            batch,
            new_left_dim,
            physical_dim,
            right_dim,
        )
        state.tensors[qubit - 1] = absorb_right_canonical_transfer(
            state.tensors[qubit - 1], transfer
        )
    state.orthogonality_center = int(target_site)
    state._canonical_center_valid = True
