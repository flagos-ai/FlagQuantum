"""Fermionic density observables and measured-tensor assembly.

Execution stays with the caller's native expectation workflow. The convention is
gamma[p, q] = <a†[p] a[q]> in physical mode order, before binary encoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

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
    _validate_density_values(values)
    return _hermitian_density(values, width)


def _hermitian_density(values: Sequence[torch.Tensor], width: int) -> torch.Tensor:
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
        [entries[p, q] for p in range(width) for q in range(width)]
    ).reshape(width, width)


def _validate_density_values(values: Sequence[torch.Tensor]) -> None:
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


def two_body_density_observables(
    *, n_modes: int, encoding: Encoding = "jordan_wigner"
) -> tuple[Observable, ...]:
    """Build Hermitian measurements for the two-body density of a full register.

    For lexicographic pairs (p, q), p < q, measure each pair diagonal followed
    by real and imaginary parts against each later pair (r, s). The measured
    monomial is a†p a†q a_s a_r. The circuit must match the width and encoding.

    Raises:
        ValueError: Fewer than two modes or an unsupported encoding.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2, dtype=torch.complex128, device="cpu").x(0).x(1)
        >>> observables = two_body_density_observables(n_modes=2)
        >>> result = fq.run(circuit, outputs=[fq.expectation(o) for o in observables])
        >>> pair = two_body_density(result.expectations, n_modes=2)[0, 1, 1, 0]
        >>> torch.testing.assert_close(pair, pair.new_tensor(1 + 0j), atol=1e-12, rtol=0)
    """
    width = count(n_modes, "n_modes", minimum=2)
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {encoding!r}")
    pairs = tuple(combinations(range(width), 2))
    creators = [create(p) * create(q) for p, q in pairs]
    annihilators = [pair.dagger() for pair in creators]
    observables: list[Observable] = []
    for i, left in enumerate(creators):
        operators = [left * annihilators[i]]
        for right in annihilators[i + 1 :]:
            term = left * right
            operators.extend(
                (0.5 * (term + term.dagger()), -0.5j * (term - term.dagger()))
            )
        observables.extend(
            to_observable(encode_fermion(op, n_modes=width, encoding=encoding).operator)
            for op in operators
        )
    return tuple(observables)


def two_body_density(values: Sequence[torch.Tensor], *, n_modes: int) -> torch.Tensor:
    """Assemble Gamma[p,q,r,s] = <a†p a†q a_r a_s> in physical mode order.

    Requires at least two modes and binomial(n_modes, 2)**2 ordered finite
    single-element CPU float32 or float64 values of one dtype, as returned by
    the corresponding measurements. Returns a dense complex tensor with four
    mode axes and preserves gradients. Enforces pair antisymmetry and Hermitian
    conjugation without projecting the data onto a physical state.

    Raises:
        TypeError: An expectation is not a supported real tensor.
        ValueError: Width, count, device, shape, dtype agreement or finiteness
            is invalid. Values cannot reveal a mislabeled encoding or order.
    """
    width = count(n_modes, "n_modes", minimum=2)
    pair_count = width * (width - 1) // 2
    if len(values) != pair_count**2:
        raise ValueError(
            "expected binomial(n_modes, 2)**2 ordered density expectations"
        )
    _validate_density_values(values)
    matrix = _hermitian_density(values, pair_count)
    p, q = torch.triu_indices(width, width, offset=1, device="cpu")
    index = torch.zeros((width, width), dtype=torch.long, device="cpu")
    index[p, q] = index[q, p] = torch.arange(pair_count, device="cpu")
    modes = torch.arange(width, device="cpu")
    sign = (modes[None, :] - modes[:, None]).sign()
    # Pair-matrix columns use reversed annihilator order.
    return -matrix[index[:, :, None, None], index[None, None, :, :]] * (
        sign[:, :, None, None] * sign[None, None, :, :]
    )
