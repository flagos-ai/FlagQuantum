"""Von Neumann entropy of a subsystem, from the state an execution left behind.

``fq.vn_entropy(qubits)`` asks one question -- how entangled are these qubits with
the rest -- and there are two families of route to it. A dense result already holds
amplitudes, so the reduced state is assembled and diagonalised. An MPS holds its
state split at every bond, so the same answer is read off the canonical form after
contracting the requested run, which never materialises the full state.

Both routes are derived from the same definition, ``S(rho) = -Tr rho log rho``, and
neither is a fast approximation of the other. They are put in one module because the
thing that must not diverge is the *definition*: two files with one logarithm each
would be two places where a tolerance or a ``log_base`` convention could drift.

The entropy is a function of the eigenvalues of a state, so its rounding error is
not the rounding error of an amplitude. ``eigvalsh`` and ``svdvals`` both return
singular values accurate to a few multiples of the unit roundoff, but ``x log x``
amplifies that near ``x = 1`` and ``log x`` amplifies it near ``x = 0``. The
tolerances the callers compare against are measured against a higher-precision
reference for that reason, and are not derived from ``torch.finfo(dtype).eps``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

# A probability below this contributes less than 1e-27 nats at complex64, which is
# below the rounding error of the sum it would be added to, so it is dropped rather
# than logged. It is a floor on the *arithmetic* and not a claim that the true
# probability is zero: a state whose spectrum is this small is one whose entropy is
# zero to the precision the dtype can represent either way.
_PROBABILITY_FLOOR = 1e-30


def log_base_divisor(log_base: float | None) -> float:
    """Return the number to divide a natural-log entropy by.

    ``None`` means nats and divides by one. PennyLane 0.45.1 divides by
    ``log(log_base)`` and validates nothing, so ``log_base=1`` returns ``inf``,
    ``log_base=-1`` returns ``nan`` and ``log_base=0`` returns the natural-log
    answer by accident of a truthiness test. Zero, negative and one are refused
    here instead: each of those is an answer the caller cannot have meant, and a
    silent ``nan`` travelling out through a result field is exactly the failure
    mode the package refuses elsewhere. The divergence from PennyLane is recorded
    in ``docs/api-changes/FQ-VN-ENTROPY-OUTPUT-20261101.md``.
    """

    if log_base is None:
        return 1.0
    if isinstance(log_base, bool) or not isinstance(log_base, (int, float)):
        raise TypeError(f"log_base must be a number, got {log_base!r}")
    value = float(log_base)
    if not math.isfinite(value):
        raise ValueError("log_base must be finite")
    if value <= 0.0:
        raise ValueError("log_base must be positive")
    if value == 1.0:
        raise ValueError("log_base must not be 1, which has no logarithm")
    return math.log(value)


def _entropy_from_sorted_probabilities(
    probabilities: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """``-sum p log p`` over the last axis of a non-negative tensor."""

    positive = probabilities > _PROBABILITY_FLOOR
    safe = torch.where(positive, probabilities, torch.ones_like(probabilities))
    terms = torch.where(positive, safe * torch.log(safe), torch.zeros_like(safe))
    entropy = -terms.sum(dim=-1)
    # Rounding can leave a state that is pure to the dtype a hair below zero. Clamping
    # here rather than accepting it keeps every caller's non-negativity check honest:
    # an entropy that is negative is a defect, and one that is negative by 1e-7 is one
    # the caller would have to know to forgive.
    return torch.clamp(entropy, min=0.0) / divisor


def entropy_from_probabilities(
    probabilities: torch.Tensor,
    *,
    log_base: float | None = None,
) -> torch.Tensor:
    """Von Neumann entropy from a spectrum, ``(batch, rank)`` or ``(rank,)``."""

    divisor = log_base_divisor(log_base)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
        return _entropy_from_sorted_probabilities(probabilities, divisor)[0]
    return _entropy_from_sorted_probabilities(probabilities, divisor)


def entropy_from_density_matrix(
    rho: torch.Tensor,
    *,
    log_base: float | None = None,
) -> torch.Tensor:
    """Von Neumann entropy of a Hermitian density matrix, ``(batch, d, d)``.

    ``eigvalsh`` is used rather than ``eigvals`` because the matrix is Hermitian by
    construction and the routine that knows that returns real, ascending values.
    Imaginary parts from a non-Hermitian input would be silently discarded by a
    real-eigenvalue routine, which is why the caller never hands one in: the only
    producers are the partial trace and the run contraction below.
    """

    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("a density matrix must be a batch of square matrices")
    eigenvalues = torch.linalg.eigvalsh(rho).clamp_min(0.0)
    return entropy_from_probabilities(eigenvalues, log_base=log_base)


def contiguous_ascending_run(
    qubits: Iterable[int],
    n_qubits: int,
) -> tuple[int, int] | None:
    """Return ``(first, last)`` if the selection is one ascending contiguous run.

    The MPS route contracts the named sites in order, so a selection that skips a
    site or names one twice is not a run and returns ``None``. Ascending is required
    rather than merely contiguous because the contraction is left to right; a
    descending run describes the same subsystem and would give the same entropy, but
    routing it here would silently reorder the operator and lose the property that
    the named order is the order used.
    """

    labels = tuple(int(qubit) for qubit in qubits)
    if not labels:
        return None
    first = labels[0]
    if first < 0 or labels[-1] != first + len(labels) - 1:
        return None
    if labels[-1] >= n_qubits:
        return None
    if any(label != first + offset for offset, label in enumerate(labels)):
        return None
    return first, labels[-1]


def mps_run_density_matrix(state: Any, first: int, last: int) -> torch.Tensor:
    """Reduced density matrix on the contiguous sites ``first..last`` of an MPS.

    The canonical center is moved to ``last`` first, which makes every site left of
    the run left-canonical and every site right of it right-canonical. That is what
    lets the two open bonds be traced instead of contracted against the rest of the
    chain: each of those environments is then exactly the identity, so summing over
    the bond index *is* the environment contraction rather than an approximation of
    it. The alternative -- building the full state and reducing it -- costs ``4 ** n``
    entries where this costs ``4 ** len(run)`` times the bond dimension.

    The state is copied first. Canonicalising moves tensors in place, and the result
    object a measurement is handed is the one ``fq.run`` returns, so mutating it would
    make a second output on the same execution depend on whether this one ran.
    """

    working = state.copy()
    working.move_orthogonality_center(last)
    batch, left_dim, physical_dim, bond = working.tensors[first].shape
    contraction = working.tensors[first]
    length = 1
    for site in range(first + 1, last + 1):
        following = working.tensors[site]
        inner = following.shape[1]
        if inner != contraction.shape[-1]:
            raise ValueError("MPS bond dimensions do not align across the run")
        contraction = torch.matmul(
            contraction.reshape(batch, left_dim * physical_dim, inner),
            following.reshape(batch, inner, -1),
        )
        physical_dim *= following.shape[2]
        contraction = contraction.reshape(
            batch, left_dim, physical_dim, following.shape[3]
        )
        length += 1
    # Contracting both open bonds leaves the operator on the run. ``length`` is not
    # used after the loop; the reshape above is what keeps the physical index
    # contiguous, so the operator's basis order is the qubit order that was named.
    return torch.einsum("blpr,blqr->bpq", contraction, contraction.conj())


def mps_prefix_spectrum(state: Any, length: int) -> torch.Tensor | None:
    """Schmidt probabilities across the cut after site ``length - 1``, or ``None``.

    In mixed canonical form with the orthogonality center at site ``j``, the state
    factors as ``|psi> = sum_l |phi_l> (x) |w_l>`` where ``|phi_l>`` spans the prefix
    up to ``j`` and is orthonormal. The coefficient matrix of that factorization is
    therefore ``isometry @ W``, and its singular values are ``W``'s: the Schmidt
    values of the ``j | j + 1`` cut are exactly the singular values of the center
    tensor read as a ``(left * physical) x right`` matrix. One ``svdvals`` of a bond
    matrix answers the question, with no reduced matrix and no full state.

    ``None`` is returned when the cut is a trivial one -- the prefix is the whole
    chain, or it is empty -- because neither has a bond to read and the caller's
    other route answers those. The state is copied for the same reason the run
    contraction copies it: moving the center rewrites tensors in place.
    """

    if length <= 0 or length >= state.n_qubits:
        return None
    working = state.copy()
    working.move_orthogonality_center(length - 1)
    tensor: torch.Tensor = working.tensors[length - 1]
    batch, left_dim, physical_dim, right_dim = tensor.shape
    matrix = tensor.reshape(batch, left_dim * physical_dim, right_dim)
    # Annotated because torch's `linalg` namespace carries no stubs under the
    # repository's type-check settings, so an inferred binding there is `Any`.
    singular_values: torch.Tensor = torch.linalg.svdvals(matrix)
    # The singular values are amplitudes, so the probabilities are their squares. This
    # is the one place the spectrum is formed; the caller's other route reaches the
    # same numbers through a partial trace, which is what makes the two an
    # independent check on each other rather than a restatement.
    return singular_values.real.square()


__all__ = (
    "contiguous_ascending_run",
    "entropy_from_density_matrix",
    "entropy_from_probabilities",
    "log_base_divisor",
    "mps_prefix_spectrum",
    "mps_run_density_matrix",
)
