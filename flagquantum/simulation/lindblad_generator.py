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

The generator is assembled from the pieces as a
:class:`flagquantum.operators.SuperOperator`, which is the sum of left and right
multiplication actions this equation is written in. Both the matrix-free action
and the dense reference are then one call on that sum, so the two cannot drift
apart: the Hamiltonian is a superoperator factor instead of a matrix, and only
:meth:`Liouvillian.dense` asks it for one.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..operators import SuperOperator
from .matrix_free_hamiltonian import PauliSum

__all__ = ("DEFAULT_DENSE_GENERATOR_BYTES", "Liouvillian")

DEFAULT_DENSE_GENERATOR_BYTES = 16 * 1024 * 1024
# Byte ceiling above which ``L`` is refused instead of materialized.


class Liouvillian:
    """The generator of a Lindblad master equation, held as its pieces.

    The Hamiltonian may be a matrix-free :class:`~.matrix_free_hamiltonian.PauliSum`
    or a dense matrix. Collapse operators are the already-normalized
    ``sqrt(rate) * operator`` factors; the constructor assembles the Hamiltonian
    commutator and every dissipator term into one
    :class:`~flagquantum.operators.SuperOperator`, and the derived products
    ``L^dag L`` appear once as shared factors of the two one-sided terms.
    """

    __slots__ = ("_adjoint", "_dimension", "_superoperator")

    def __init__(
        self,
        hamiltonian: PauliSum | torch.Tensor,
        collapse: Sequence[torch.Tensor],
        *,
        hilbert_dimension: int,
    ) -> None:
        self._dimension = int(hilbert_dimension)
        generator = SuperOperator.left_multiply(hamiltonian) * (-1j)
        generator += SuperOperator.right_multiply(hamiltonian) * (1j)
        # The Hilbert--Schmidt adjoint of the generator differs from it in
        # exactly two places: its Hamiltonian part changes sign, because the
        # adjoint of ``-i [H, .]`` is ``+i [H, .]``, and the dissipator's
        # sandwich is applied the other way round, because the adjoint of
        # ``rho -> L rho L^dag`` is ``sigma -> L^dag sigma L``. The
        # anticommutator is its own adjoint, since ``L^dag L`` is Hermitian.
        # Both are accumulated in one loop so that a change to the generator
        # cannot silently miss the adjoint that differentiates it.
        adjoint = SuperOperator.left_multiply(hamiltonian) * (1j)
        adjoint += SuperOperator.right_multiply(hamiltonian) * (-1j)
        for operator in collapse:
            dagger = torch.conj(operator).T
            product = dagger @ operator
            generator += SuperOperator.left_right_multiply(operator, dagger)
            generator += SuperOperator.left_multiply(product) * (-0.5)
            generator += SuperOperator.right_multiply(product) * (-0.5)
            adjoint += SuperOperator.left_right_multiply(dagger, operator)
            adjoint += SuperOperator.left_multiply(product) * (-0.5)
            adjoint += SuperOperator.right_multiply(product) * (-0.5)
        self._superoperator = generator
        self._adjoint = adjoint

    @property
    def hilbert_dimension(self) -> int:
        """Return ``2**n``, the side of the state this generator acts on."""

        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype every application and the dense form use."""

        return self._superoperator.dtype

    def derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``d rho / dt`` for a density matrix, materializing nothing.

        This is the generator applied through the density form. It is the same
        operator the vectorized matrix represents, which is why the matrix-free
        exponential never forms the ``16**n`` superoperator.
        """

        return self._superoperator.apply(state)

    def adjoint_derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``L^dag sigma``, the generator read through the adjoint.

        The adjoint is taken with respect to the Hilbert--Schmidt inner product
        ``<sigma, L rho> = trace(sigma^dag L rho)``, which is the inner product a
        real cost over density matrices differentiates through. Applying it is
        what propagates a cost gradient backwards across one output interval;
        it is the same superoperator algebra as :meth:`derivative`, with the
        Hamiltonian part sign-flipped and the dissipator's sandwich reversed.
        """

        return self._adjoint.apply(state)

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
        return self._superoperator.dense(max_bytes=int(max_bytes))
