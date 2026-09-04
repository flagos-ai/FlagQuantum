"""Local statevector entry backed by the current native implementation.

This is the Simulation-owned migration seam used by Runtime today. The adapter
can disappear once the numerical loop moves out of :class:`Circuit`; callers
should continue using this function rather than depending on that legacy home.
"""

from __future__ import annotations

import torch

from ..core.ir import CircuitIR


def run_local_statevector(
    program: CircuitIR,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute validated IR on one resolved device without Runtime policy."""

    if not isinstance(program, CircuitIR):
        raise TypeError("program must be a CircuitIR")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from ..circuit import Circuit

    return Circuit.from_ir(
        program,
        bsz=batch_size,
        device=device,
        dtype=dtype,
    ).state()


__all__ = ("run_local_statevector",)
