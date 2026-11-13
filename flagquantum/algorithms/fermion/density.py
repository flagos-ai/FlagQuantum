"""Fermionic one-body density observables and measured-tensor assembly.

Execution stays with the caller's native expectation workflow. The convention is
gamma[p, q] = <a†[p] a[q]> in physical mode order, before binary encoding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...core.algebra.pauli import count
from ...observables import Observable, annihilate, create, number
from .encoding import ENCODINGS, Encoding, encode_fermion
from .observables import to_observable


def one_body_density_observables(
    *, n_modes: int, encoding: Encoding = "jordan_wigner"
) -> tuple[Observable, ...]:
    """Build n_modes**2 Hermitian measurements on a full encoded register.

    For each p, return n[p], followed by real and imaginary parts of a†p aq
    for increasing q>p. The circuit must use the same width and encoding.
    Tapered registers and automatic state-basis conversion are unsupported.

    Raises:
        ValueError: The width is not positive or the encoding is unsupported.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2, dtype=torch.complex128, device="cpu").x(0)
        >>> observables = one_body_density_observables(n_modes=2)
        >>> values = fq.run(circuit, outputs=[fq.expectation(o) for o in observables])
        >>> one_body_density(values.expectations, n_modes=2).diag().real.tolist()
        [1.0, 0.0]
    """
    width = count(n_modes, "n_modes", minimum=1)
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {encoding!r}")
    observables: list[Observable] = []
    for p in range(width):
        operators = [number(p)]
        for q in range(p + 1, width):
            hopping = create(p) * annihilate(q)
            operators.extend(
                [
                    0.5 * (hopping + hopping.dagger()),
                    -0.5j * (hopping - hopping.dagger()),
                ]
            )
        observables.extend(
            to_observable(encode_fermion(op, n_modes=width, encoding=encoding).operator)
            for op in operators
        )
    return tuple(observables)


def one_body_density(values: Sequence[torch.Tensor], *, n_modes: int) -> torch.Tensor:
    """Assemble gamma from the ordered real density-observable expectations.

    Returns a CPU complex64/complex128 tensor of shape (n_modes, n_modes),
    preserving circuit-parameter gradients. This enforces conjugate symmetry,
    without projecting, clipping or normalizing measured values.

    Args:
        values: Finite single-element CPU float32/float64 tensors of a common dtype,
            in the order returned by one_body_density_observables.
        n_modes: Full physical register width used to construct the observables.

    Raises:
        TypeError: An expectation is not a supported real tensor.
        ValueError: Width, count, shape, device, dtype agreement or finiteness
            is invalid. Values cannot reveal a mislabeled encoding or order.
    """
    width = count(n_modes, "n_modes", minimum=1)
    if len(values) != width * width:
        raise ValueError("expected n_modes**2 ordered density expectation values")
    dtype: torch.dtype | None = None
    for value in values:
        if not isinstance(value, torch.Tensor) or value.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError("density expectations must be real float32/float64 tensors")
        if value.numel() != 1 or value.device.type != "cpu":
            raise ValueError("density expectations must be single-element CPU tensors")
        if dtype is not None and value.dtype != dtype:
            raise ValueError("density expectations must share one dtype")
        if not bool(torch.isfinite(value)):
            raise ValueError("density expectations must be finite")
        dtype = value.dtype
    entries: dict[tuple[int, int], torch.Tensor] = {}
    position = 0
    for p in range(width):
        diagonal = values[position].reshape(())
        entries[p, p] = torch.complex(diagonal, torch.zeros_like(diagonal))
        position += 1
        for q in range(p + 1, width):
            value = torch.complex(
                values[position].reshape(()), values[position + 1].reshape(())
            )
            entries[p, q], entries[q, p] = value, value.conj()
            position += 2
    return torch.stack(
        [torch.stack([entries[p, q] for q in range(width)]) for p in range(width)]
    )
