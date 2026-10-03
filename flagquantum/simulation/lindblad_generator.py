"""The Lindblad generator in its vectorized, superoperator form.

The master equation is linear in the density matrix, so it is also an ordinary
linear ODE on the vectorized state,

    d|rho>> / dt = L|rho>>,

where ``L`` is the Liouvillian. The vectorization identity that produces ``L``
is ``vec(A rho B) = (A (x) B^T) vec(rho)`` for the row-major vectorization
``Tensor.reshape(-1)`` this module uses, so the Hamiltonian part becomes
``-i (H (x) I - I (x) H^T)`` and each collapse operator contributes
``L (x) conj(L) - (L^dag L (x) I + I (x) (L^dag L)^T) / 2``.

Two facts make this representation worth having without ever paying for it.
First, ``L`` has ``16**n`` entries, so :meth:`Liouvillian.dense` refuses to
materialize it above a declared ceiling and :meth:`Liouvillian.derivative`
applies the same generator through the density form instead -- which is exactly
the vectorization identity read the other way. Second, the conventional
properties of a Liouvillian are statements about that matrix, and the unit suite
checks them against this representation, which is what turns the algebra above
into verified behavior rather than a comment. Trace preservation is
``L^T vec(I) = 0``. Hermiticity preservation is that ``L`` commutes with the
antiunitary involution ``v -> T conj(v)``, where ``T`` transposes the matrix
form; commutation with ``T`` alone would be false, since a generator with a real
Hamiltonian satisfies ``T L T = -L``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .matrix_free_hamiltonian import PauliSum

__all__ = ("DEFAULT_DENSE_GENERATOR_BYTES", "Liouvillian")

DEFAULT_DENSE_GENERATOR_BYTES = 16 * 1024 * 1024
"""Byte ceiling above which ``L`` is refused instead of materialized."""


class Liouvillian:
    """The generator of a Lindblad master equation, held as its pieces.

    The Hamiltonian may be a matrix-free :class:`~.matrix_free_hamiltonian.PauliSum`
    or a dense matrix. Collapse operators are the already-normalized
    ``sqrt(rate) * operator`` factors, and their products are derived once so
    that every application reuses them.
    """

    __slots__ = ("_collapse", "_dimension", "_hamiltonian", "_products")

    def __init__(
        self,
        hamiltonian: PauliSum | torch.Tensor,
        collapse: Sequence[torch.Tensor],
        *,
        dimension: int,
    ) -> None:
        self._dimension = int(dimension)
        self._hamiltonian = hamiltonian
        self._collapse = tuple(collapse)
        self._products = tuple(
            torch.conj(operator).T @ operator for operator in self._collapse
        )

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension ``2**n`` of the generator."""

        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        if isinstance(self._hamiltonian, PauliSum):
            return self._hamiltonian._dtype
        return self._hamiltonian.dtype

    def derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``d rho / dt`` for a density matrix, materializing nothing.

        This is the generator applied through the density form. It is the same
        operator the vectorized matrix represents, which is why the matrix-free
        exponential never forms the ``16**n`` superoperator.
        """

        if isinstance(self._hamiltonian, PauliSum):
            value = -1j * self._hamiltonian.commutator(state)
        else:
            value = -1j * (self._hamiltonian @ state - state @ self._hamiltonian)
        for operator, product in zip(
            self._collapse, self._products, strict=True
        ):
            value = value + operator @ state @ torch.conj(operator).T
            value = value - 0.5 * (product @ state + state @ product)
        return value

    def dense(self, *, max_bytes: int = DEFAULT_DENSE_GENERATOR_BYTES) -> torch.Tensor:
        """Return the ``4**n x 4**n`` vectorized generator.

        The form is refused above ``max_bytes`` rather than allocated, because
        the entry count is ``(2**n)**4``: the representation is useful for small
        systems, as an independent reference for the matrix-free action, and as
        the exact generator of a dense matrix exponential, but it is never a
        path for production sizes. The default ceiling admits up to ``n = 5``,
        where the generator is ``1024 x 1024`` and exactly fills 16 MiB, and
        refuses ``n = 6`` and above.

        Raises:
            ValueError: If the materialized generator would exceed ``max_bytes``.
        """

        dimension = self._dimension
        side = dimension * dimension
        entries = side * side
        element_bytes = self.dtype.itemsize
        # The sum enforces the same ceiling with the same arithmetic, so this
        # refusal never fires second. It is stated here because the message is
        # part of this method's contract and names the generator, which is what
        # a caller reading it is holding.
        if entries * element_bytes > int(max_bytes):
            raise ValueError(
                f"a {dimension}-dimensional vectorized generator is {side} x {side} "
                f"with {entries} entries ({entries * element_bytes} bytes), above "
                f"the {int(max_bytes)}-byte ceiling; apply the generator instead of "
                "materializing it"
            )
        dtype = self.dtype
        device = (
            self._hamiltonian._device
            if isinstance(self._hamiltonian, PauliSum)
            else self._hamiltonian.device
        )
        identity = torch.eye(dimension, dtype=dtype, device=device)
        if isinstance(self._hamiltonian, PauliSum):
            hamiltonian = self._hamiltonian.dense()
        else:
            hamiltonian = self._hamiltonian
        generator = -1j * (
            torch.kron(hamiltonian, identity) - torch.kron(identity, hamiltonian.T)
        )
        for operator, product in zip(self._collapse, self._products, strict=True):
            generator = generator + torch.kron(operator, torch.conj(operator))
            generator = generator - 0.5 * (
                torch.kron(product, identity) + torch.kron(identity, product.T)
            )
        return generator
