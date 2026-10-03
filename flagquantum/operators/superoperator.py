"""A finite sum of left and right multiplication actions on a matrix.

A superoperator is a linear map on the space of matrices, written as a sum of
terms

    coefficient * left @ state @ right

where a missing ``left`` means the term acts only from the right and a missing
``right`` means it acts only from the left. Both are present in the Lindblad
dissipator's ``L rho L^dag`` term; only one is present in ``-i H rho`` and in the
``-1/2 {L^dag L, rho}`` anticommutator terms. The algebra exists so that a
generator can be assembled from those actions, inspected, and applied without
writing a density-form expression by hand.

The factors are *matrix operators*, not necessarily dense matrices. A factor
must provide ``dtype``, ``dimension`` (for a ``torch.Tensor`` these are ``dtype``
and ``shape[0]``), ``__matmul__`` and ``__rmatmul__`` with a square state, and a
``dense()`` matrix. ``torch.Tensor`` is the implementation that exists today;
:class:`flagquantum.simulation.matrix_free_hamiltonian.PauliSum` already exposes
``dtype``, ``dimension`` and ``dense()``, and the two-sided action it needs is
what its ``commutator`` already computes, so a matrix-free ``H`` can join a term
without being materialized once it satisfies the remaining two members. Until
then this module claims no matrix-free factor.

The dense form uses the row-major vectorization the rest of the repository uses,
``vec(A rho B) = (A (x) B^T) vec(rho)``, so the right factor is transposed and
never conjugate-transposed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import torch

__all__ = ("DEFAULT_DENSE_MATRIX_BYTES", "SuperOperator")

DEFAULT_DENSE_MATRIX_BYTES = 16 * 1024 * 1024
# Byte ceiling above which :meth:`SuperOperator.dense` refuses to allocate.
#
# The dense form is ``(2**n)**4`` entries, so it is a reference for small systems and
# the exact generator of a dense matrix exponential, never a production path. The
# default ceiling admits a four-wire map and refuses the next size up.


class OperatorFactor(Protocol):
    """The matrix-operator contract a superoperator term is built from.

    A factor acts on a square matrix from one side. ``torch.Tensor`` provides
    every member directly apart from ``dimension``, which is read from the shape.
    """

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def dimension(self) -> int: ...

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor: ...

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor: ...

    def dense(self) -> torch.Tensor: ...


Factor = Any
# A superoperator factor: a ``torch.Tensor`` or an :class:`OperatorFactor`.

_Term = tuple[complex, Factor | None, Factor | None]


def _factor_dtype(operator: object) -> torch.dtype:
    dtype = getattr(operator, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"an operator factor must expose a torch dtype, got {operator!r}"
        )
    if not dtype.is_complex:
        raise ValueError(f"an operator factor must have a complex dtype, got {dtype}")
    return dtype


def _factor_dimension(operator: object) -> int:
    if isinstance(operator, torch.Tensor):
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(
                "a tensor operator factor must be a square matrix, got shape "
                f"{tuple(operator.shape)}"
            )
        return int(operator.shape[0])
    for name in ("__matmul__", "__rmatmul__", "dense"):
        if not callable(getattr(operator, name, None)):
            raise ValueError(
                f"an operator factor must support {name} with a square state"
            )
    dimension = getattr(operator, "dimension", None)
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError(
            "an operator factor must declare a positive integer dimension, got "
            f"{dimension!r}"
        )
    return dimension


def _dense_matrix(operator: Factor) -> torch.Tensor:
    if isinstance(operator, torch.Tensor):
        return operator
    matrix = operator.dense()
    if not isinstance(matrix, torch.Tensor):
        raise ValueError(
            f"an operator factor must return a tensor from dense(), got {matrix!r}"
        )
    return matrix


def _coefficient(value: object) -> complex:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ValueError(f"a superoperator coefficient must be a scalar, got {value!r}")
    return complex(value)


def _same_factor(left: Factor | None, right: Factor | None) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    return bool(left == right)


class SuperOperator:
    """A sum of left and right multiplication actions on a matrix.

    The instance is constructed empty and grown with ``+=``, which appends
    terms, exactly as the CUDA-Q superoperator accumulates terms. ``+`` returns
    a new instance with the concatenated terms, scalar multiplication scales
    every coefficient, and equality compares the term sequence structurally.
    Every term names at least one side, because each of the three constructors
    supplies one; a term that multiplies from neither side is unrepresentable.

    Examples:
        >>> import torch
        >>> from flagquantum.operators import SuperOperator
        >>> lowering = torch.tensor(
        ...     [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
        ... )
        >>> number = lowering.conj().T @ lowering
        >>> decay = SuperOperator.left_right_multiply(lowering, lowering.conj().T)
        >>> decay += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
        >>> decay += SuperOperator.right_multiply((-0.5 + 0.0j) * number)
        >>> excited = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
        >>> decay.apply(excited).tolist()
        [[(1+0j), 0j], [0j, (-1+0j)]]
    """

    __slots__ = ("_dimension", "_dtype", "_terms")

    def __init__(self) -> None:
        self._terms: list[_Term] = []
        self._dimension: int | None = None
        self._dtype: torch.dtype | None = None

    @classmethod
    def _from_terms(cls, terms: list[_Term]) -> SuperOperator:
        instance = cls()
        instance._terms = list(terms)
        instance._resolve_summary()
        return instance

    @classmethod
    def left_multiply(cls, operator: Factor) -> SuperOperator:
        """Return the map ``rho -> operator @ rho``."""

        return cls._from_terms([(1.0 + 0.0j, operator, None)])

    @classmethod
    def right_multiply(cls, operator: Factor) -> SuperOperator:
        """Return the map ``rho -> rho @ operator``."""

        return cls._from_terms([(1.0 + 0.0j, None, operator)])

    @classmethod
    def left_right_multiply(cls, left: Factor, right: Factor) -> SuperOperator:
        """Return the map ``rho -> left @ rho @ right``."""

        return cls._from_terms([(1.0 + 0.0j, left, right)])

    @property
    def terms(self) -> tuple[_Term, ...]:
        """Return the ``(coefficient, left, right)`` terms in insertion order."""

        return tuple(self._terms)

    def _resolve_summary(self) -> None:
        dimensions: set[int] = set()
        dtypes: set[torch.dtype] = set()
        for _, left, right in self._terms:
            for factor in (left, right):
                if factor is None:
                    continue
                dimensions.add(_factor_dimension(factor))
                dtypes.add(_factor_dtype(factor))
        if len(dimensions) > 1:
            raise ValueError(
                "every superoperator factor must share one dimension, got "
                f"{sorted(dimensions)}"
            )
        if len(dtypes) > 1:
            raise ValueError(
                "every superoperator factor must share one dtype, got "
                f"{sorted(str(dtype) for dtype in dtypes)}"
            )
        self._dimension = next(iter(dimensions), None)
        self._dtype = next(iter(dtypes), None)

    def _require_summary(self) -> tuple[int, torch.dtype]:
        if self._dimension is None or self._dtype is None:
            raise ValueError(
                "an empty superoperator has no dimension or dtype; accumulate at "
                "least one term before inspecting or applying it"
            )
        return self._dimension, self._dtype

    @property
    def dimension(self) -> int:
        """Return the side length of the matrices the map acts on."""

        return self._require_summary()[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype every term and every view uses."""

        return self._require_summary()[1]

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self) -> Iterator[_Term]:
        """Iterate over the ``(coefficient, left, right)`` terms."""

        return iter(self._terms)

    def __iadd__(self, other: SuperOperator) -> SuperOperator:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        appended = len(other._terms)
        self._terms = self._terms + other._terms
        try:
            self._resolve_summary()
        except ValueError:
            self._terms = self._terms[: len(self._terms) - appended]
            raise
        return self

    def __add__(self, other: SuperOperator) -> SuperOperator:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        return self._from_terms(self._terms + other._terms)

    def __mul__(self, coefficient: complex) -> SuperOperator:
        scalar = _coefficient(coefficient)
        return self._from_terms(
            [(scalar * value, left, right) for value, left, right in self._terms]
        )

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        if self._dimension != other._dimension or self._dtype != other._dtype:
            return False
        if len(self._terms) != len(other._terms):
            return False
        return all(
            mine[0] == theirs[0]
            and _same_factor(mine[1], theirs[1])
            and _same_factor(mine[2], theirs[2])
            for mine, theirs in zip(self._terms, other._terms, strict=True)
        )

    __hash__ = None  # type: ignore[assignment]

    def _validate_state(self, state: torch.Tensor) -> None:
        dimension, dtype = self._require_summary()
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"a superoperator state must be a tensor, got {state!r}")
        if state.dtype != dtype:
            raise ValueError(
                "the state dtype must match the superoperator dtype: "
                f"{state.dtype} != {dtype}"
            )
        shape = tuple(int(size) for size in state.shape)
        expected: tuple[int, ...]
        if state.ndim == 2:
            expected = (dimension, dimension)
        elif state.ndim == 3:
            expected = (shape[0], dimension, dimension)
        else:
            raise ValueError(
                f"the state must have shape ({dimension}, {dimension}) or "
                f"(batch, {dimension}, {dimension}), got {shape}"
            )
        if shape != expected:
            raise ValueError(
                f"the state must have shape ({dimension}, {dimension}) or "
                f"(batch, {dimension}, {dimension}), got {shape}"
            )

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Return the map applied to a ``(d, d)`` or ``(batch, d, d)`` state.

        Raises:
            ValueError: If the state shape or dtype does not match the map.
        """

        self._validate_state(state)
        total = torch.zeros_like(state)
        for coefficient, left, right in self._terms:
            term = state
            if left is not None:
                term = (
                    left @ term
                    if isinstance(left, torch.Tensor)
                    else left.__matmul__(term)
                )
            if right is not None:
                term = (
                    term @ right
                    if isinstance(right, torch.Tensor)
                    else right.__rmatmul__(term)
                )
            total = total + coefficient * term
        return total

    def dense(self, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES) -> torch.Tensor:
        """Return the row-major ``d**2 x d**2`` matrix of the map.

        The form is refused above ``max_bytes`` rather than allocated, because
        the entry count is ``(2**n)**4``, so this is a small-system reference and
        a test oracle rather than a production path.

        Raises:
            ValueError: If the materialized matrix would exceed ``max_bytes``.
        """

        dimension, dtype = self._require_summary()
        side = dimension * dimension
        entries = side * side
        element_bytes = dtype.itemsize
        if entries * element_bytes > int(max_bytes):
            raise ValueError(
                f"a {dimension}-dimensional superoperator is {side} x {side} with "
                f"{entries} entries ({entries * element_bytes} bytes), above the "
                f"{int(max_bytes)}-byte ceiling; apply the map instead of "
                "materializing it"
            )
        identity = torch.eye(dimension, dtype=dtype)
        matrix = torch.zeros((side, side), dtype=dtype)
        for coefficient, left, right in self._terms:
            if left is None:
                block = torch.kron(
                    identity, _dense_matrix(right).transpose(0, 1).contiguous()
                )
            elif right is None:
                block = torch.kron(_dense_matrix(left), identity)
            else:
                block = torch.kron(
                    _dense_matrix(left),
                    _dense_matrix(right).transpose(0, 1).contiguous(),
                )
            matrix = matrix + coefficient * block
        return matrix

    def __repr__(self) -> str:
        return (
            f"SuperOperator(terms={len(self._terms)}, dimension={self._dimension}, "
            f"dtype={self._dtype})"
        )
