"""Backend-neutral memory estimates used by runtime planning."""

from __future__ import annotations


def estimate_state_bytes(n_wires: int, bsz: int = 1, complex_bytes: int = 8) -> int:
    """Estimate dense statevector memory."""

    return int(bsz) * (1 << int(n_wires)) * int(complex_bytes)


def estimate_density_bytes(n_wires: int, bsz: int = 1, complex_bytes: int = 8) -> int:
    """Estimate dense density-matrix memory."""

    dim = 1 << int(n_wires)
    return int(bsz) * dim * dim * int(complex_bytes)


def estimate_mps_bytes(
    n_wires: int,
    *,
    bsz: int = 1,
    max_bond: int | None = None,
    complex_bytes: int = 8,
) -> int:
    """Estimate MPS memory for a fixed maximum bond dimension."""

    n_wires = int(n_wires)
    if max_bond is None:
        max_bond = 2 ** max(0, n_wires // 2)
    return int(bsz) * n_wires * 2 * int(max_bond) * int(max_bond) * int(complex_bytes)


def estimate_tensor_network_bytes(
    n_wires: int, bsz: int = 1, complex_bytes: int = 8
) -> int:
    """Estimate output-state memory for a tensor-network contraction."""

    return estimate_state_bytes(n_wires, bsz=bsz, complex_bytes=complex_bytes)


__all__ = [
    "estimate_density_bytes",
    "estimate_mps_bytes",
    "estimate_state_bytes",
    "estimate_tensor_network_bytes",
]
