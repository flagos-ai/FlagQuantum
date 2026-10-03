"""Pauli-sum Hamiltonians evaluated without materializing a dense matrix.

Continuous-time evolution needs ``[H, rho]`` and a Hermiticity decision, never
the Hamiltonian matrix itself.  One Pauli string costs a bit mask, a sign mask
and a complex coefficient here, while the dense form costs ``4**n`` complex
entries: at thirteen wires that is 1024 MiB of Hamiltonian against a few
kilobytes of terms, and the dense form also has to be built term by term before
any of it can be used.

The representation is exact.  A Pauli string acts on an amplitude index ``j``
as ``P|j> = (-1)**popcount(j & signs) |j ^ mask>``, so both ``P @ state`` and
``state @ P`` are a gather on the row or column axis followed by a sign
multiply, and ``H`` then satisfies the matrix-operator contract a superoperator
factor needs without ever becoming a matrix.  Wire 0 is the most significant
amplitude bit, matching ``expand_operator`` in
:mod:`flagquantum.simulation.density_matrix`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch

DEFAULT_COMMUTATOR_BLOCK_BYTES = 64 * 1024 * 1024
# Soft byte budget for the temporaries one blocked Hamiltonian action allocates.


@dataclass(frozen=True, slots=True)
class PauliSumTerm:
    """One term: the coefficient of a single Pauli string.

    ``coefficient`` is the coefficient the caller wrote, so a real coefficient
    always denotes a Hermitian string.  ``mask`` holds the amplitude bit
    indices that carry an ``X`` or ``Y`` factor, and ``signs`` the ascending
    indices that carry a ``Z`` or ``Y`` factor.  A ``Y`` factor appears in
    both, because ``Y = i X Z``: the ``i`` is applied from the overlap rather
    than stored, which keeps the coefficient physical and makes Hermiticity a
    statement about real coefficients.
    """

    coefficient: complex
    mask: int = 0
    signs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "mask", int(self.mask))
        object.__setattr__(
            self, "signs", tuple(sorted(int(sign) for sign in self.signs))
        )

    def action_coefficient(self) -> complex:
        """Return the coefficient of the bit-flip word this term executes.

        The word is ``X(mask) @ Z(signs)``, which is Hermitian only when the
        two masks do not overlap; each overlapping bit contributes the ``i``
        that turns that word back into the Pauli string the caller named.
        """

        sign_bits = 0
        for sign in self.signs:
            sign_bits |= 1 << sign
        return self.coefficient * (1j ** (self.mask & sign_bits).bit_count())


def _sign_vector(
    indices: torch.Tensor,
    signs: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(-1) ** popcount(index & signs)`` for every index."""

    if not signs:
        return torch.ones(indices.numel(), dtype=dtype, device=device)
    parity = torch.zeros(indices.numel(), dtype=torch.int64, device=device)
    for sign in signs:
        parity = parity ^ ((indices >> sign) & 1)
    return (1 - 2 * parity).to(dtype)


class PauliSum:
    """A Hamiltonian as an ordered sum of Pauli terms, held matrix-free.

    Construction derives the bit masks and sign vectors every later operation
    reuses, so an evolution pays for them once rather than once per stage.
    """

    __slots__ = (
        "_coefficients",
        "_device",
        "_dimension",
        "_dtype",
        "_keys",
        "_n_wires",
        "_permutations",
        "_scales",
        "_terms",
    )

    def __init__(
        self,
        terms: Sequence[PauliSumTerm],
        *,
        n_wires: int,
        dimension: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._n_wires = int(n_wires)
        self._dimension = int(dimension)
        self._dtype = dtype
        self._device = torch.device(device)
        self._terms = tuple(terms)
        keys = [(term.mask, term.signs) for term in self._terms]
        distinct = list(dict.fromkeys(keys))
        positions = {key: index for index, key in enumerate(distinct)}
        self._keys = tuple(positions[key] for key in keys)
        indices = torch.arange(self._dimension, dtype=torch.int64, device=self._device)
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self._permutations = tuple(indices ^ mask for mask, _ in distinct)
        self._scales = tuple(
            _sign_vector(indices, signs, dtype=real_dtype, device=self._device)
            for _, signs in distinct
        )
        self._coefficients = tuple(
            torch.tensor(
                term.action_coefficient(), dtype=self._dtype, device=self._device
            )
            for term in self._terms
        )

    @property
    def n_wires(self) -> int:
        return self._n_wires

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def terms(self) -> tuple[PauliSumTerm, ...]:
        return self._terms

    def commutator(
        self,
        state: torch.Tensor,
        *,
        block_bytes: int = DEFAULT_COMMUTATOR_BLOCK_BYTES,
    ) -> torch.Tensor:
        """Return ``H @ state - state @ H`` for a density matrix or a batch.

        A two-dimensional ``state`` is one density matrix; a three-dimensional
        one carries a leading batch axis, so every member is advanced under the
        same Hamiltonian. The row axis is processed in blocks whose temporaries
        stay inside ``block_bytes``, so the working set does not grow with
        ``4**n`` while the result and the state keep their full size.
        """

        self._require_applicable(state)
        result = torch.empty_like(state)
        for start, stop in self._blocks(state, block_bytes):
            result[..., start:stop, :] = self._commutator_block(state, start, stop)
        return result

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``H @ state`` without materializing ``H``.

        This is the left factor of the commutator exposed on its own, because a
        dissipator needs one side at a time. The row axis is blocked exactly as
        the commutator blocks it, so the temporary working set is the same.
        """

        return self._one_sided(state, block=self._left_block)

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``state @ H`` without materializing ``H``."""

        return self._one_sided(state, block=self._right_block)

    def _one_sided(
        self,
        state: torch.Tensor,
        *,
        block: Callable[[torch.Tensor, int, int], torch.Tensor],
    ) -> torch.Tensor:
        self._require_applicable(state)
        result = torch.empty_like(state)
        for start, stop in self._blocks(state, DEFAULT_COMMUTATOR_BLOCK_BYTES):
            result[..., start:stop, :] = block(state, start, stop)
        return result

    def _require_applicable(self, state: torch.Tensor) -> None:
        """Refuse a state whose dtype, device, or shape this action cannot take."""

        if state.dtype != self._dtype or state.device != self._device:
            raise ValueError(
                "the state dtype and device must match the planned Hamiltonian"
            )
        if state.ndim not in (2, 3) or tuple(state.shape[-2:]) != (
            self._dimension,
            self._dimension,
        ):
            raise ValueError(
                f"the state must have shape ({self._dimension}, {self._dimension}) "
                f"or (batch, {self._dimension}, {self._dimension})"
            )

    def _blocks(
        self, state: torch.Tensor, block_bytes: int
    ) -> Iterator[tuple[int, int]]:
        """Yield the row ranges whose temporaries fit inside ``block_bytes``.

        The budget is read at call time so a caller can lower it, and the
        arithmetic counts the three same-sized temporaries one block allocates
        once per batch member.
        """

        per_row = 1 if state.ndim == 2 else int(state.shape[0])
        per_row *= 3 * self._dimension * state.element_size()
        rows = max(1, min(self._dimension, int(block_bytes) // per_row))
        for start in range(0, self._dimension, rows):
            yield start, min(start + rows, self._dimension)

    def _left_block(self, state: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Return rows ``start:stop`` of ``H @ state``."""

        if state.ndim == 2:
            total = torch.zeros(
                (stop - start, self._dimension), dtype=self._dtype, device=self._device
            )
            for coefficient, key in zip(self._coefficients, self._keys, strict=True):
                permutation = self._permutations[key]
                scale = self._scales[key]
                sources = permutation[start:stop]
                total = total + coefficient * (
                    state.index_select(0, sources)
                    * scale.index_select(0, sources).unsqueeze(1)
                )
            return total
        total = torch.zeros(
            (state.shape[0], stop - start, self._dimension),
            dtype=self._dtype,
            device=self._device,
        )
        for coefficient, key in zip(self._coefficients, self._keys, strict=True):
            permutation = self._permutations[key]
            scale = self._scales[key]
            sources = permutation[start:stop]
            total = total + coefficient * (
                state.index_select(1, sources)
                * scale.index_select(0, sources).reshape(1, -1, 1)
            )
        return total

    def _right_block(self, state: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Return rows ``start:stop`` of ``state @ H``."""

        if state.ndim == 2:
            block = state[start:stop]
            total = torch.zeros(
                (stop - start, self._dimension), dtype=self._dtype, device=self._device
            )
            for coefficient, key in zip(self._coefficients, self._keys, strict=True):
                permutation = self._permutations[key]
                scale = self._scales[key]
                total = total + coefficient * (
                    block.index_select(1, permutation) * scale.unsqueeze(0)
                )
            return total
        block = state[:, start:stop]
        total = torch.zeros(
            (block.shape[0], stop - start, self._dimension),
            dtype=self._dtype,
            device=self._device,
        )
        for coefficient, key in zip(self._coefficients, self._keys, strict=True):
            permutation = self._permutations[key]
            scale = self._scales[key]
            total = total + coefficient * (
                block.index_select(2, permutation) * scale.reshape(1, 1, -1)
            )
        return total

    def _commutator_block(
        self, state: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        if state.ndim == 2:
            block = state[start:stop]
            total = torch.zeros(
                (stop - start, self._dimension), dtype=self._dtype, device=self._device
            )
            for coefficient, key in zip(self._coefficients, self._keys, strict=True):
                permutation = self._permutations[key]
                scale = self._scales[key]
                sources = permutation[start:stop]
                from_left = state.index_select(0, sources) * scale.index_select(
                    0, sources
                ).unsqueeze(1)
                from_right = block.index_select(1, permutation) * scale.unsqueeze(0)
                total = total + coefficient * (from_left - from_right)
            return total
        block = state[:, start:stop]
        total = torch.zeros(
            (block.shape[0], stop - start, self._dimension),
            dtype=self._dtype,
            device=self._device,
        )
        for coefficient, key in zip(self._coefficients, self._keys, strict=True):
            permutation = self._permutations[key]
            scale = self._scales[key]
            sources = permutation[start:stop]
            from_left = state.index_select(1, sources) * scale.index_select(
                0, sources
            ).reshape(1, -1, 1)
            from_right = block.index_select(2, permutation) * scale.reshape(1, 1, -1)
            total = total + coefficient * (from_left - from_right)
        return total

    def dense(self) -> torch.Tensor:
        """Return the dense matrix, for consumers that must serialize ``H``.

        This is the only operation here that costs ``4**n`` entries; it exists
        because the serialized Lindblad request still stores a dense
        Hamiltonian.
        """

        indices = torch.arange(self._dimension, dtype=torch.int64, device=self._device)
        matrix = torch.zeros(
            (self._dimension, self._dimension), dtype=self._dtype, device=self._device
        )
        for term in self._terms:
            columns = indices ^ term.mask
            values = _sign_vector(
                columns, term.signs, dtype=self._dtype, device=self._device
            )
            matrix[indices, columns] += term.action_coefficient() * values
        return matrix

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliSum):
            return NotImplemented
        return (
            self._n_wires == other._n_wires
            and self._dimension == other._dimension
            and self._dtype == other._dtype
            and self._terms == other._terms
        )

    def __hash__(self) -> int:
        return hash((self._n_wires, self._dimension, self._dtype, self._terms))

    def __repr__(self) -> str:
        return f"PauliSum(n_wires={self._n_wires}, terms={len(self._terms)})"


__all__ = ("DEFAULT_COMMUTATOR_BLOCK_BYTES", "PauliSum", "PauliSumTerm")
